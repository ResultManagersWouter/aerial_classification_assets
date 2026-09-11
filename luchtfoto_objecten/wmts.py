from __future__ import annotations

import io
import logging
import math
import re
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from rasterio.transform import Affine
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASIS_URL = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0"
# De infraroodopnamen staan bij PDOK op een eigen service. Laagnamen eindigen daar op IR.
BASIS_URL_INFRAROOD = "https://service.pdok.nl/hwh/luchtfotocir/wmts/v1_0"


def is_infrarood(laag: str) -> bool:
    return laag.endswith("IR")


def basis_url(laag: str) -> str:
    return BASIS_URL_INFRAROOD if is_infrarood(laag) else BASIS_URL
TEGELMATRIXSET = "EPSG:28992"
TEGELGROOTTE = 256
RD_LINKSBOVEN = (-285401.92, 903401.92)
SCHAALNOEMER_NIVEAU_NUL = 12_288_000.0
METER_PER_EENHEID = 0.00028
MAX_TEGELS = 40_000

STANDAARDLAAG = "Actueel_orthoHR"
# Terugval als de capabilities niet opgehaald kunnen worden.
BEKENDE_LAGEN = {
    "Actueel_orthoHR": "Luchtfoto Actueel Ortho 8cm RGB",
    "Actueel_ortho25": "Luchtfoto Actueel Ortho 25cm RGB",
    "2026_orthoHR": "Luchtfoto 2026 Ortho 8cm RGB",
    "2025_orthoHR": "Luchtfoto 2025 Ortho 8cm RGB",
    "2024_orthoHR": "Luchtfoto 2024 Ortho 8cm RGB",
}
JAARLAAG_PATROON = re.compile(r"^(\d{4})_(quick)?ortho(HR|25)(IR)?$")


def haal_beschikbare_lagen(cache_map: Path | None = None, timeout: int = 60) -> dict[str, str]:
    """Leest de laagnamen rechtstreeks uit de capabilities, zodat nieuwe jaargangen vanzelf meekomen."""
    lagen: dict[str, str] = {}
    for dienst, url in (("rgb", BASIS_URL), ("cir", BASIS_URL_INFRAROOD)):
        lagen.update(_lagen_van_dienst(dienst, url, cache_map, timeout))
    return lagen or dict(BEKENDE_LAGEN)


def _lagen_van_dienst(dienst: str, basis: str, cache_map: Path | None, timeout: int) -> dict[str, str]:
    cachepad = cache_map / f"wmts_capabilities_{dienst}.xml" if cache_map else None
    inhoud = None
    if cachepad is not None and cachepad.exists():
        inhoud = cachepad.read_text(encoding="utf-8")
    if inhoud is None:
        try:
            antwoord = requests.get(
                f"{basis}/wmts",
                params={"request": "GetCapabilities", "service": "wmts"},
                timeout=timeout,
                headers={"User-Agent": "luchtfoto-objecten-identificatie/0.1"},
            )
            antwoord.raise_for_status()
            inhoud = antwoord.text
            if cachepad is not None:
                cachepad.parent.mkdir(parents=True, exist_ok=True)
                cachepad.write_text(inhoud, encoding="utf-8")
        except Exception as fout:
            logger.warning("Capabilities van %s niet op te halen (%s)", dienst, fout)
            return dict(BEKENDE_LAGEN) if dienst == "rgb" else {}

    naamruimten = {"wmts": "http://www.opengis.net/wmts/1.0", "ows": "http://www.opengis.net/ows/1.1"}
    lagen = {}
    for laag in ElementTree.fromstring(inhoud).iter(f"{{{naamruimten['wmts']}}}Layer"):
        identificatie = laag.find(f"{{{naamruimten['ows']}}}Identifier")
        titel = laag.find(f"{{{naamruimten['ows']}}}Title")
        if identificatie is not None and identificatie.text:
            lagen[identificatie.text] = titel.text if titel is not None else ""
    return lagen


def jaarlagen_nieuwste_eerst(lagen: dict[str, str]) -> list[str]:
    """Jaargangen van nieuw naar oud, binnen een jaar eerst 8cm en dan 25cm."""
    gevonden = []
    for identificatie in lagen:
        overeenkomst = JAARLAAG_PATROON.match(identificatie)
        if overeenkomst:
            jaar = int(overeenkomst.group(1))
            is_quick = bool(overeenkomst.group(2))
            resolutierang = 0 if overeenkomst.group(3) == "HR" else 1
            gevonden.append(((-jaar, resolutierang, is_quick), identificatie))
    return [identificatie for _, identificatie in sorted(gevonden)]


def resolutie(zoom: int) -> float:
    """Meters per pixel op een zoomniveau van de RD-tegelmatrix."""
    return (SCHAALNOEMER_NIVEAU_NUL / (2**zoom)) * METER_PER_EENHEID


def tegelspanwijdte(zoom: int) -> float:
    return resolutie(zoom) * TEGELGROOTTE


def tegelbereik(bbox: tuple[float, float, float, float], zoom: int) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox
    span = tegelspanwijdte(zoom)
    kolom_min = math.floor((xmin - RD_LINKSBOVEN[0]) / span)
    kolom_max = math.floor((xmax - RD_LINKSBOVEN[0]) / span)
    rij_min = math.floor((RD_LINKSBOVEN[1] - ymax) / span)
    rij_max = math.floor((RD_LINKSBOVEN[1] - ymin) / span)
    return kolom_min, rij_min, kolom_max, rij_max


@dataclass
class Uitsnede:
    afbeelding: np.ndarray
    transform: Affine
    bbox: tuple[float, float, float, float]
    zoom: int
    laag: str

    @property
    def resolutie_m(self) -> float:
        return abs(self.transform.a)


class LuchtfotoWMTS:
    """Haalt tegels op bij de landelijke voorziening Beeldmateriaal via PDOK."""

    def __init__(
        self,
        laag: str = STANDAARDLAAG,
        zoom: int = 15,
        cache_map: Path | None = None,
        max_werkers: int = 4,
        timeout: int = 60,
    ) -> None:
        self.laag = laag
        self.zoom = zoom
        self.cache_map = cache_map
        self.max_werkers = max_werkers
        self.timeout = timeout
        self.sessie = requests.Session()
        self.sessie.headers.update({"User-Agent": "luchtfoto-objecten-identificatie/0.1"})
        herhaling = Retry(
            total=4,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.sessie.mount("https://", HTTPAdapter(max_retries=herhaling, pool_maxsize=max_werkers * 2))

    def tegel_url(self, kolom: int, rij: int) -> str:
        return f"{basis_url(self.laag)}/{self.laag}/{TEGELMATRIXSET}/{self.zoom:02d}/{kolom}/{rij}.jpeg"

    def _cachepad(self, kolom: int, rij: int) -> Path | None:
        if self.cache_map is None:
            return None
        return self.cache_map / "wmts" / self.laag / f"{self.zoom:02d}" / str(kolom) / f"{rij}.jpeg"

    def haal_tegel(self, kolom: int, rij: int) -> np.ndarray:
        pad = self._cachepad(kolom, rij)
        if pad is not None and pad.exists():
            with Image.open(pad) as afbeelding:
                return np.asarray(afbeelding.convert("RGB"))

        antwoord = self.sessie.get(self.tegel_url(kolom, rij), timeout=self.timeout)
        if antwoord.status_code == 404:
            return np.zeros((TEGELGROOTTE, TEGELGROOTTE, 3), dtype=np.uint8)
        antwoord.raise_for_status()

        if pad is not None:
            pad.parent.mkdir(parents=True, exist_ok=True)
            pad.write_bytes(antwoord.content)
        with Image.open(io.BytesIO(antwoord.content)) as afbeelding:
            return np.asarray(afbeelding.convert("RGB"))

    def haal_uitsnede(self, bbox: tuple[float, float, float, float], toon_voortgang: bool = True) -> Uitsnede:
        kolom_min, rij_min, kolom_max, rij_max = tegelbereik(bbox, self.zoom)
        aantal_kolommen = kolom_max - kolom_min + 1
        aantal_rijen = rij_max - rij_min + 1
        aantal_tegels = aantal_kolommen * aantal_rijen
        if aantal_tegels > MAX_TEGELS:
            raise ValueError(
                f"Dit gebied vraagt {aantal_tegels} tegels op zoom {self.zoom}. "
                f"Kies een kleiner gebied of een lager zoomniveau."
            )
        logger.info(
            "Ophalen %s tegels (%s x %s) voor %s op zoom %s",
            aantal_tegels, aantal_kolommen, aantal_rijen, self.laag, self.zoom,
        )

        mozaiek = np.zeros((aantal_rijen * TEGELGROOTTE, aantal_kolommen * TEGELGROOTTE, 3), dtype=np.uint8)
        opdrachten = [(k, r) for r in range(rij_min, rij_max + 1) for k in range(kolom_min, kolom_max + 1)]

        def verwerk(opdracht: tuple[int, int]) -> tuple[int, int, np.ndarray]:
            kolom, rij = opdracht
            return kolom, rij, self.haal_tegel(kolom, rij)

        voortgang = _voortgangsbalk(len(opdrachten), toon_voortgang)
        with ThreadPoolExecutor(max_workers=self.max_werkers) as pool:
            for kolom, rij, tegel in pool.map(verwerk, opdrachten):
                y0 = (rij - rij_min) * TEGELGROOTTE
                x0 = (kolom - kolom_min) * TEGELGROOTTE
                mozaiek[y0:y0 + TEGELGROOTTE, x0:x0 + TEGELGROOTTE] = tegel
                if voortgang is not None:
                    voortgang.update(1)
        if voortgang is not None:
            voortgang.close()

        res = resolutie(self.zoom)
        span = tegelspanwijdte(self.zoom)
        mozaiek_x0 = RD_LINKSBOVEN[0] + kolom_min * span
        mozaiek_y0 = RD_LINKSBOVEN[1] - rij_min * span

        xmin, ymin, xmax, ymax = bbox
        kolom_start = int(round((xmin - mozaiek_x0) / res))
        kolom_eind = int(round((xmax - mozaiek_x0) / res))
        rij_start = int(round((mozaiek_y0 - ymax) / res))
        rij_eind = int(round((mozaiek_y0 - ymin) / res))
        uitsnede = mozaiek[rij_start:rij_eind, kolom_start:kolom_eind]

        transform = Affine(res, 0.0, mozaiek_x0 + kolom_start * res, 0.0, -res, mozaiek_y0 - rij_start * res)
        werkelijke_bbox = (
            transform.c,
            transform.f - uitsnede.shape[0] * res,
            transform.c + uitsnede.shape[1] * res,
            transform.f,
        )
        return Uitsnede(afbeelding=uitsnede, transform=transform, bbox=werkelijke_bbox, zoom=self.zoom, laag=self.laag)


def _voortgangsbalk(totaal: int, tonen: bool):
    if not tonen:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=totaal, unit="tegel", desc="luchtfoto")
