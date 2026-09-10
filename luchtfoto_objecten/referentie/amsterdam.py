from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import box
from urllib3.util.retry import Retry

from luchtfoto_objecten.geo_hulp import RD

logger = logging.getLogger(__name__)

BASIS_URL = "https://api.data.amsterdam.nl/v1"
PAGINA_GROOTTE = 1000
MAX_PAGINAS = 200


@dataclass(frozen=True)
class Registratiebron:
    sleutel: str
    pad: str
    omschrijving: str
    geometrieveld: str = "geometrie"
    extra_params: dict = field(default_factory=dict)


ACTUEEL = {"eindRegistratie[isnull]": "true"}

BRONNEN: dict[str, Registratiebron] = {
    "groenobjecten": Registratiebron(
        sleutel="groenobjecten",
        pad="objectenopenbareruimte/groenobjecten",
        omschrijving="Beheerde groenvlakken uit de objectenregistratie openbare ruimte",
    ),
    "verhardingen": Registratiebron(
        sleutel="verhardingen",
        pad="objectenopenbareruimte/verhardingen",
        omschrijving="Beheerde verhardingsvlakken, inclusief voetpad, rijbaan, parkeervak en berm",
    ),
    "parkeervakken": Registratiebron(
        sleutel="parkeervakken",
        pad="parkeervakken/parkeervakken",
        omschrijving="Parkeervakken met type, soort en regime",
        geometrieveld="geometry",
    ),
    "begroeideterreindelen": Registratiebron(
        sleutel="begroeideterreindelen",
        pad="bgt/begroeideterreindelen",
        omschrijving="Begroeide terreindelen uit de BGT, de topografische tegenhanger van groen",
        extra_params=ACTUEEL,
    ),
    "onbegroeideterreindelen": Registratiebron(
        sleutel="onbegroeideterreindelen",
        pad="bgt/onbegroeideterreindelen",
        omschrijving="Onbegroeide terreindelen uit de BGT: erf, open en gesloten verharding, zand",
        extra_params=ACTUEEL,
    ),
    "wegdelen": Registratiebron(
        sleutel="wegdelen",
        pad="bgt/wegdelen",
        omschrijving="Wegdelen uit de BGT: rijbaan, voetpad, fietspad en parkeervlak",
        extra_params=ACTUEEL,
    ),
    "beheerkaart": Registratiebron(
        sleutel="beheerkaart",
        pad="beheerkaart/basis/kaart",
        omschrijving="BGT-vlakken op percelen in eigendom van Amsterdam, het beheergebied",
    ),
    "panden": Registratiebron(
        sleutel="panden",
        pad="bgt/panden",
        omschrijving="Panden uit de BGT, gebruikt om daken uit de beeldanalyse te houden",
        extra_params=ACTUEEL,
    ),
    "waterdelen": Registratiebron(
        sleutel="waterdelen",
        pad="bgt/waterdelen",
        omschrijving="Waterdelen uit de BGT, gebruikt als uitsluitmasker",
        extra_params=ACTUEEL,
    ),
}


class AmsterdamRegistratie:
    """Leest de open objectregistraties van de gemeente Amsterdam in Rijksdriehoek."""

    def __init__(self, cache_map: Path | None = None, timeout: int = 90, gebruik_cache: bool = True) -> None:
        self.cache_map = cache_map
        self.timeout = timeout
        self.gebruik_cache = gebruik_cache and cache_map is not None
        self.sessie = requests.Session()
        self.sessie.headers.update({
            "Accept-Crs": "EPSG:28992",
            "Content-Crs": "EPSG:28992",
            "User-Agent": "luchtfoto-objecten-identificatie/0.1",
        })
        herhaling = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        self.sessie.mount("https://", HTTPAdapter(max_retries=herhaling))

    def haal(self, sleutel: str, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        if sleutel not in BRONNEN:
            raise KeyError(f"Onbekende bron '{sleutel}'. Beschikbaar: {', '.join(sorted(BRONNEN))}")
        bron = BRONNEN[sleutel]
        cachepad = self._cachepad(bron, bbox)
        if cachepad is not None and cachepad.exists():
            logger.info("Registratie %s uit cache", sleutel)
            return self._naar_gdf(json.loads(cachepad.read_text(encoding="utf-8")), bbox)

        kenmerken = self._haal_alle_paginas(bron, bbox)
        verzameling = {"type": "FeatureCollection", "features": kenmerken}
        if cachepad is not None:
            cachepad.parent.mkdir(parents=True, exist_ok=True)
            cachepad.write_text(json.dumps(verzameling), encoding="utf-8")
        logger.info("Registratie %s opgehaald: %s objecten", sleutel, len(kenmerken))
        return self._naar_gdf(verzameling, bbox)

    def haal_meerdere(self, sleutels: list[str], bbox: tuple[float, float, float, float]) -> dict[str, gpd.GeoDataFrame]:
        return {sleutel: self.haal(sleutel, bbox) for sleutel in sleutels}

    def _haal_alle_paginas(self, bron: Registratiebron, bbox: tuple[float, float, float, float]) -> list[dict]:
        xmin, ymin, xmax, ymax = bbox
        wkt = f"POLYGON(({xmin} {ymin},{xmax} {ymin},{xmax} {ymax},{xmin} {ymax},{xmin} {ymin}))"
        params = {
            "_format": "geojson",
            "_pageSize": PAGINA_GROOTTE,
            f"{bron.geometrieveld}[intersects]": wkt,
            **bron.extra_params,
        }
        url = f"{BASIS_URL}/{bron.pad}/"
        kenmerken: list[dict] = []
        for _ in range(MAX_PAGINAS):
            antwoord = self.sessie.get(url, params=params, timeout=self.timeout)
            antwoord.raise_for_status()
            inhoud = antwoord.json()
            kenmerken.extend(inhoud.get("features", []))
            volgende = _volgende_pagina(inhoud)
            if not volgende:
                return kenmerken
            url, params = volgende, None
        logger.warning("Maximum aantal pagina's bereikt voor %s", bron.sleutel)
        return kenmerken

    def _cachepad(self, bron: Registratiebron, bbox: tuple[float, float, float, float]) -> Path | None:
        if not self.gebruik_cache or self.cache_map is None:
            return None
        sleutel = json.dumps([bron.pad, sorted(bron.extra_params.items()), [round(getal, 1) for getal in bbox]])
        vingerafdruk = hashlib.sha1(sleutel.encode("utf-8")).hexdigest()[:16]
        return self.cache_map / "referentie" / f"{bron.sleutel}_{vingerafdruk}.geojson"

    @staticmethod
    def _naar_gdf(verzameling: dict, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        kenmerken = verzameling.get("features", [])
        if not kenmerken:
            return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], dtype="geometry")}, geometry="geometry", crs=RD)
        gdf = gpd.GeoDataFrame.from_features(kenmerken, crs=RD)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        ongeldig = ~gdf.geometry.is_valid
        if ongeldig.any():
            gdf.loc[ongeldig, "geometry"] = gdf.loc[ongeldig, "geometry"].make_valid()
        return gpd.clip(gdf, box(*bbox)).reset_index(drop=True)


def _volgende_pagina(inhoud: dict) -> str | None:
    links = inhoud.get("_links")
    if isinstance(links, list):
        for link in links:
            if link.get("rel") == "next" and link.get("href"):
                return link["href"]
    if isinstance(links, dict):
        volgende = links.get("next")
        if isinstance(volgende, dict):
            return volgende.get("href")
        if isinstance(volgende, str):
            return volgende
    return None
