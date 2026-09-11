"""Panden uit de BAG, als vangnet tegen groen op daken.

Het hoogtemodel hoort gebouwen er al uit te houden, maar dat gaat mis zodra het AHN daar
geen maaiveld heeft: onder een pand is geen grondmeting, dus het DTM zit er vol gaten.
Bij de ArenA is 45 procent van het maaiveldmodel leeg, precies waar de bebouwing staat.
Die gaten worden nu opgevuld vanaf de rand, maar een tweede slot is het waard, want een
groen dak dat als gazon in de beheerregistratie belandt is een dure fout.

De BAG is een landelijke basisregistratie en open via de WFS van PDOK, dus dit blijft
binnen dezelfde soort bronnen als de rest: geen gemeentelijke assetregistratie, gewoon
topografie.

Let op dat de BAG het pand als grondvlak geeft, niet het dak. Een overstekend dak of een
balkon steekt daar overheen, vandaar de buffer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import box
from urllib3.util.retry import Retry

from luchtfoto_objecten.geo_hulp import RD

logger = logging.getLogger(__name__)

WFS_URL = "https://service.pdok.nl/lv/bag/wfs/v2_0"
MAX_PANDEN = 10_000


def haal_panden(bbox, cache_map: Path | None = None, timeout: int = 120) -> gpd.GeoDataFrame:
    """Alle BAG-panden die het gebied raken, in RD."""
    sleutel = hashlib.sha1(json.dumps([round(getal, 1) for getal in bbox]).encode()).hexdigest()[:16]
    cachepad = cache_map / "bag" / f"panden_{sleutel}.geojson" if cache_map else None
    if cachepad is not None and cachepad.exists():
        return _naar_gdf(json.loads(cachepad.read_text(encoding="utf-8")), bbox)

    sessie = requests.Session()
    sessie.headers.update({"User-Agent": "luchtfoto-objecten-identificatie/0.1"})
    herhaling = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    sessie.mount("https://", HTTPAdapter(max_retries=herhaling))
    try:
        antwoord = sessie.get(
            WFS_URL,
            params={
                "service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": "bag:pand",
                "outputFormat": "application/json", "srsName": "EPSG:28992",
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:28992",
                "count": MAX_PANDEN,
            },
            timeout=timeout,
        )
        antwoord.raise_for_status()
        inhoud = antwoord.json()
    except Exception as fout:
        logger.warning("BAG-panden niet op te halen (%s), we vertrouwen op het hoogtemodel", fout)
        return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], dtype="geometry")}, geometry="geometry", crs=RD)

    if cachepad is not None:
        cachepad.parent.mkdir(parents=True, exist_ok=True)
        cachepad.write_text(json.dumps(inhoud), encoding="utf-8")
    panden = _naar_gdf(inhoud, bbox)
    logger.info("BAG: %s panden in dit gebied", len(panden))
    return panden


def _naar_gdf(inhoud: dict, bbox) -> gpd.GeoDataFrame:
    kenmerken = inhoud.get("features", [])
    if not kenmerken:
        return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], dtype="geometry")}, geometry="geometry", crs=RD)
    panden = gpd.GeoDataFrame.from_features(kenmerken, crs=RD)
    panden = panden[~panden.geometry.is_empty & panden.geometry.notna()].copy()
    ongeldig = ~panden.geometry.is_valid
    if ongeldig.any():
        panden.loc[ongeldig, "geometry"] = panden.loc[ongeldig, "geometry"].make_valid()
    return gpd.clip(panden, box(*bbox)).reset_index(drop=True)


def pandmasker(bbox, vorm, transform, cache_map: Path | None = None, buffer_m: float = 1.0):
    """Masker van alles wat pand is, met een buffer voor overstekende daken."""
    from luchtfoto_objecten.raster import polygonen_naar_masker

    panden = haal_panden(bbox, cache_map)
    if panden.empty:
        import numpy as np

        return np.zeros(vorm, dtype=bool)
    return polygonen_naar_masker(panden.geometry, vorm, transform, buffer_m=buffer_m)
