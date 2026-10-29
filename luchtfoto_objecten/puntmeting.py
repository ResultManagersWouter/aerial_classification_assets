"""Meten rond een punt: vegetatiegetal uit de luchtfoto en hoogte uit het AHN.

Bedoeld voor puntobjecten uit een register, zoals bomen en stobbes, waar je per punt wilt
weten wat de open bronnen op die plek laten zien. Een heel gebied classificeren doet
classificatie.py; dit kijkt alleen in een klein venster rond elk punt.

De metingen per punt worden bewaard in de assetmap (afgeleid/<doel>/), met de plek in het
punt-id. Een tweede run meet alleen wat er nieuw is.
"""

from __future__ import annotations

import io
import logging
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from luchtfoto_objecten import opslag
from luchtfoto_objecten.wmts import RD_LINKSBOVEN, TEGELGROOTTE, LuchtfotoWMTS, resolutie

logger = logging.getLogger(__name__)

# Een venster zonder enige meting (water, rand van de dekking) levert NaN op, dat is bedoeld
warnings.filterwarnings("ignore", category=RuntimeWarning, message="(All-NaN|Mean of empty|Degrees of freedom)")

ZOOM = 14                # 21 cm per pixel; de 25cm-opnamen hebben niet meer detail
STRAAL_KERN_M = 1.5      # rond de stam
STRAAL_KROON_M = 3.0     # waarbinnen kroonvolume en kroonbedekking geteld worden
RING_M = (4.0, 7.0)      # de omgeving, om een kroon te onderscheiden van een groen veld
KROONHOOGTE_M = 2.5      # vanaf hier telt objecthoogte als kroon; een stobbe of auto blijft eronder
AHN_CEL_M = 250.0

NDVI_KENMERKEN = ["ndvi_kern", "ndvi_kroon", "ndvi_ring", "ndvi_contrast"]
AHN_KENMERKEN = ["hoogte_max", "hoogte_mediaan", "hoogte_max_3m", "ruwheid", "kroonvolume_m3", "kroonbedekking"]


def punt_ids(punten, voorvoegsel: str) -> list[str]:
    """Een id met de plek erin, zodat een bewaarde meting niet hergebruikt wordt als het punt verschoof."""
    return [f"{voorvoegsel}_{i}_{p.x:.1f}_{p.y:.1f}" for i, p in zip(punten["id"], punten.geometry)]


def afstanden(n: int, pixel_m: float, dkolom: float, drij: float) -> np.ndarray:
    """Afstand in meters van elk pixelmidden in een venster van 2n+1 tot het punt.

    dkolom en drij zijn de afstand in pixels van het middelste pixelmidden tot het punt.
    """
    stappen = np.arange(-n, n + 1, dtype=np.float64)
    return pixel_m * np.hypot(stappen[None, :] + dkolom, stappen[:, None] + drij)


# ---------------------------------------------------------------- vegetatiegetal


class Tegelbron:
    """Leest vensters rond punten uit de WMTS-tegels van één laag.

    Tegels komen uit de tegelset in de assetmap; wat daar niet in staat wordt bij PDOK
    opgehaald en erbij gezet. Gedecodeerde tegels blijven in het geheugen zolang ze vaak
    nodig zijn.
    """

    def __init__(self, laag: str, zoom: int = ZOOM, werkers: int = 8) -> None:
        self.laag, self.zoom, self.werkers = laag, zoom, werkers
        self.wmts = LuchtfotoWMTS(laag=laag, zoom=zoom, cache_map=None, max_werkers=werkers)
        self.opslag = opslag.Tegelopslag(laag)
        oude_cache = opslag.PROJECT / "data" / "cache" / "wmts" / laag
        if oude_cache.exists():
            overgenomen = self.opslag.importeer_map(oude_cache, zoom)
            if overgenomen:
                logger.info("%s: %s tegels uit de oude bestandscache overgenomen", laag, overgenomen)
        self.res = resolutie(zoom)
        self.tegel = lru_cache(maxsize=4096)(self._tegel)

    def _pixel(self, x: float, y: float) -> tuple[float, float]:
        return (x - RD_LINKSBOVEN[0]) / self.res, (RD_LINKSBOVEN[1] - y) / self.res

    def tegels_voor(self, x: float, y: float, straal: float) -> set[tuple[int, int]]:
        kolom, rij = self._pixel(x, y)
        n = straal / self.res
        return {(k, r)
                for k in range(int((kolom - n) // TEGELGROOTTE), int((kolom + n) // TEGELGROOTTE) + 1)
                for r in range(int((rij - n) // TEGELGROOTTE), int((rij + n) // TEGELGROOTTE) + 1)}

    def _download(self, tegel: tuple[int, int]) -> bool:
        try:
            antwoord = self.wmts.sessie.get(self.wmts.tegel_url(*tegel), timeout=60)
            if antwoord.status_code == 404:
                return True
            antwoord.raise_for_status()
            self.opslag.schrijf(self.zoom, *tegel, antwoord.content)
            return True
        except Exception as fout:
            logger.warning("Tegel %s van %s niet op te halen: %s", tegel, self.laag, fout)
            return False

    def vooraf_ophalen(self, tegels: set[tuple[int, int]]) -> None:
        """Haalt de tegels die nog niet in de tegelset staan parallel binnen."""
        nodig = sorted(set(tegels) - self.opslag.aanwezig(self.zoom))
        if nodig:
            logger.info("%s: %s van %s tegels ophalen bij PDOK", self.laag, f"{len(nodig):,}", f"{len(tegels):,}")
            with ThreadPoolExecutor(self.werkers) as pool:
                gelukt = list(tqdm(pool.map(self._download, nodig), total=len(nodig), unit="tegel", desc=self.laag))
            if not all(gelukt):
                logger.warning("%s tegels ontbreken; die punten krijgen voor deze opname geen meting",
                               len(gelukt) - sum(gelukt))
        self.opslag.bewaar()

    def _tegel(self, kolom: int, rij: int) -> np.ndarray:
        inhoud = self.opslag.lees(self.zoom, kolom, rij)
        if inhoud is None:
            return np.zeros((TEGELGROOTTE, TEGELGROOTTE, 3), dtype=np.uint8)
        try:
            with Image.open(io.BytesIO(inhoud)) as beeld:
                return np.asarray(beeld.convert("RGB"))
        except OSError:
            # Een kapotte tegel opnieuw ophalen; lukt dat niet, dan telt hij als ontbrekend
            if self._download((kolom, rij)):
                inhoud = self.opslag.lees(self.zoom, kolom, rij)
                try:
                    with Image.open(io.BytesIO(inhoud)) as beeld:
                        return np.asarray(beeld.convert("RGB"))
                except (OSError, TypeError):
                    pass
            return np.zeros((TEGELGROOTTE, TEGELGROOTTE, 3), dtype=np.uint8)

    def venster(self, x: float, y: float, straal: float) -> tuple[np.ndarray, np.ndarray]:
        """Beeld en afstand tot het punt, voor een vierkant venster van 2*straal."""
        kolom_f, rij_f = self._pixel(x, y)
        n = math.ceil(straal / self.res)
        k0, r0 = int(kolom_f) - n, int(rij_f) - n
        grootte = 2 * n + 1
        beeld = np.zeros((grootte, grootte, 3), dtype=np.uint8)
        for tk in range(k0 // TEGELGROOTTE, (k0 + grootte - 1) // TEGELGROOTTE + 1):
            for tr in range(r0 // TEGELGROOTTE, (r0 + grootte - 1) // TEGELGROOTTE + 1):
                tegel = self.tegel(tk, tr)
                bk0, br0 = tk * TEGELGROOTTE, tr * TEGELGROOTTE
                a0, a1 = max(k0, bk0), min(k0 + grootte, bk0 + TEGELGROOTTE)
                b0, b1 = max(r0, br0), min(r0 + grootte, br0 + TEGELGROOTTE)
                beeld[b0 - r0:b1 - r0, a0 - k0:a1 - k0] = tegel[b0 - br0:b1 - br0, a0 - bk0:a1 - bk0]
        return beeld, afstanden(n, self.res, int(kolom_f) + 0.5 - kolom_f, int(rij_f) + 0.5 - rij_f)


def meet_vegetatiegetal(punten, laag: str) -> pd.DataFrame:
    """NDVI rond elk punt op één infraroodopname.

    Band 1 is nabij-infrarood, band 2 rood (gemeten, zie infrarood.py). Wit is geen
    dekking en zwart een ontbrekende tegel; die pixels tellen niet mee.
    """
    bron = Tegelbron(laag)
    xs, ys = punten.geometry.x.values, punten.geometry.y.values
    tegels = set().union(*(bron.tegels_voor(x, y, RING_M[1]) for x, y in zip(xs, ys)))
    bron.vooraf_ophalen(tegels)

    # Op tegelvolgorde langslopen, dan blijven de gedecodeerde tegels in het geheugen
    volgorde = np.lexsort((ys, xs // (bron.res * TEGELGROOTTE)))
    rijen = []
    for i in tqdm(volgorde, unit="punt", desc=f"NDVI {laag}"):
        beeld, afstand = bron.venster(xs[i], ys[i], RING_M[1])
        beeld = beeld.astype(np.float32)
        nir, rood = beeld[..., 0], beeld[..., 1]
        ongeldig = (beeld.min(axis=2) > 250) | (beeld.max(axis=2) == 0)
        ndvi = np.where(ongeldig, np.nan, (nir - rood) / np.maximum(nir + rood, 1.0))
        kern = ndvi[afstand <= STRAAL_KERN_M]
        if not np.isfinite(kern).any():
            continue
        ring = ndvi[(afstand >= RING_M[0]) & (afstand <= RING_M[1])]
        kern_med = float(np.nanmedian(kern))
        ring_med = float(np.nanmedian(ring)) if np.isfinite(ring).any() else np.nan
        rijen.append({
            "punt_id": punten["punt_id"].iloc[i],
            "ndvi_kern": kern_med,
            "ndvi_kroon": float(np.nanmean(ndvi[afstand <= STRAAL_KROON_M])),
            "ndvi_ring": ring_med,
            "ndvi_contrast": kern_med - ring_med,
            # Een luchtfoto is geen echte orthofoto: een kroon helt een paar meter weg van de
            # stam. Het groenste deel binnen 3 m vangt dat beter dan de mediaan bij de stam.
            "ndvi_p90_kroon": float(np.nanpercentile(ndvi[afstand <= STRAAL_KROON_M], 90)),
        })
    return pd.DataFrame(rijen, columns=["punt_id"] + NDVI_KENMERKEN + ["ndvi_p90_kroon"])


# ---------------------------------------------------------------------------- AHN


def _ahn_cel(versie: int, cel, punten: pd.DataFrame) -> list[dict]:
    marge = STRAAL_KROON_M + 2
    xmin, ymin = cel[0] * AHN_CEL_M - marge, cel[1] * AHN_CEL_M - marge
    gelezen = opslag.objecthoogte_venster(versie, (xmin, ymin, xmin + AHN_CEL_M + 2 * marge,
                                                   ymin + AHN_CEL_M + 2 * marge))
    if gelezen is None:
        return []
    hoogte, transform = gelezen
    pixel = transform.a
    n = math.ceil(STRAAL_KROON_M / pixel)
    uit = []
    for punt in punten.itertuples():
        kolom_f = (punt.x - transform.c) / pixel
        rij_f = (transform.f - punt.y) / pixel
        kolom, rij = int(kolom_f), int(rij_f)
        venster = hoogte[rij - n:rij + n + 1, kolom - n:kolom + n + 1]
        if venster.shape != (2 * n + 1, 2 * n + 1) or not np.isfinite(venster).any():
            continue
        afstand = afstanden(n, pixel, kolom + 0.5 - kolom_f, rij + 0.5 - rij_f)
        kern, kroon = afstand <= STRAAL_KERN_M, afstand <= STRAAL_KROON_M
        kroonhoogte = np.where(kroon & (venster > KROONHOOGTE_M), venster, np.nan)
        p = f"ahn{versie}_"
        uit.append({
            "punt_id": punt.punt_id,
            p + "hoogte_max": float(np.nanmax(venster[kern])),
            p + "hoogte_mediaan": float(np.nanmedian(venster[kern])),
            p + "hoogte_max_3m": float(np.nanmax(venster[kroon])),
            p + "ruwheid": float(np.nanstd(venster[kern])),
            p + "kroonvolume_m3": float(np.nansum(kroonhoogte) * pixel ** 2),
            p + "kroonbedekking": float(np.isfinite(kroonhoogte).sum() / kroon.sum()),
        })
    return uit


def meet_ahn(punten, versie: int, werkers: int = 6) -> pd.DataFrame:
    """Hoogte, ruwheid en kroonvolume rond elk punt, uit de opgeslagen AHN-tegels.

    Kroonvolume is de objecthoogte opgeteld over alle pixels boven 2,5 m binnen 3 m van de
    stam, maal het pixeloppervlak: het volume tussen maaiveld en bovenkant kroon.
    """
    tabel = pd.DataFrame({"punt_id": punten["punt_id"].values, "x": punten.geometry.x.values,
                          "y": punten.geometry.y.values})
    opslag.zorg_voor_ahn(versie, opslag.kilometertegels(tabel["x"], tabel["y"], marge=10))
    tabel["cel"] = list(zip((tabel["x"] // AHN_CEL_M).astype(int), (tabel["y"] // AHN_CEL_M).astype(int)))
    groepen = list(tabel.groupby("cel"))
    rijen = []
    with ThreadPoolExecutor(werkers) as pool:
        for uitkomst in tqdm(pool.map(lambda g: _ahn_cel(versie, *g), groepen), total=len(groepen),
                             unit="cel", desc=f"AHN{versie}"):
            rijen.extend(uitkomst)
    return pd.DataFrame(rijen, columns=["punt_id"] + [f"ahn{versie}_{k}" for k in AHN_KENMERKEN])


# -------------------------------------------------------------------- bewaren


def bewaard(doel: str, naam: str, punten, meet, opnieuw: bool = False, kolommen: list[str] | None = None) -> pd.DataFrame:
    """Metingen per punt uit de assetmap (afgeleid/<doel>/<naam>.csv), en alleen nieuwe punten meten.

    `kolommen`: metingen die er moeten zijn. Ontbreekt zo'n kolom in het bewaarde bestand
    (een nieuwe maat), dan worden de gevraagde punten opnieuw gemeten; de andere punten in
    het bestand blijven staan. Ook punten zonder meting worden onthouden, anders wordt er
    bij elke run opnieuw naar gezocht.
    """
    map_ = opslag.map_in_assets("afgeleid", doel)
    pad = map_ / f"{naam}.csv"
    bekend = pd.read_csv(pad) if pad.exists() else pd.DataFrame(columns=["punt_id"])
    if opnieuw or any(k not in bekend.columns for k in (kolommen or [])):
        te_meten = punten
    else:
        te_meten = punten[~punten["punt_id"].isin(bekend["punt_id"])]
    if len(te_meten):
        gemeten = te_meten[["punt_id"]].merge(meet(te_meten), on="punt_id", how="left")
        bekend = pd.concat([bekend[~bekend["punt_id"].isin(te_meten["punt_id"])], gemeten], ignore_index=True)
        tijdelijk = opslag.LOKAAL_WERK / f"{doel}_{naam}.csv"
        tijdelijk.parent.mkdir(parents=True, exist_ok=True)
        bekend.to_csv(tijdelijk, index=False)
        opslag.veilig_kopieren(tijdelijk, pad)
        tijdelijk.unlink()
    return bekend[bekend["punt_id"].isin(punten["punt_id"])]
