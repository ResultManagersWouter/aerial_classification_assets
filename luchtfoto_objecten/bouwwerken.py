"""Panden uit de BGT, om gebouwen van het maaiveld te scheiden.

Het hoogtemodel hoort dit al te doen, maar het gaat mis waar het AHN geen maaiveld heeft:
onder een pand komt geen laserpuls op de grond, dus het DTM zit daar vol gaten. Die gaten
worden elders opgevuld, en dit is het tweede slot.

Waarom de BGT en niet de BAG: de BGT heeft `relatieve_hoogteligging` en de BAG niet, en
dat veld is hier precies wat je nodig hebt.

    -1 en lager   ligt ónder het maaiveld
     0            ligt op het maaiveld
     1 en hoger   ligt erboven, zoals een viaduct of een luchtbrug

Een ondergronds pand telt hier niet mee. Boven een parkeergarage of een metrostation ligt
gewoon maaiveld, vaak met gras erop, en dat als gebouw wegstrepen zou echt groen laten
verdwijnen. Op de Dam staan drie van zulke panden, bij de ArenA twaalf overige bouwwerken
op niveau 3.

De BGT is een landelijke basisregistratie en open via de OGC API van PDOK, dus dit blijft
dezelfde soort bron als de rest: topografie, geen gemeentelijke assetregistratie.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import box
from urllib3.util.retry import Retry

from luchtfoto_objecten.geo_hulp import RD

logger = logging.getLogger(__name__)

BASIS_URL = "https://api.pdok.nl/lv/bgt/ogc/v1/collections"
RD_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"
COLLECTIES = ("pand", "overigbouwwerk")
MAX_OBJECTEN = 2000


def _sessie() -> requests.Session:
    sessie = requests.Session()
    sessie.headers.update({
        "User-Agent": "luchtfoto-objecten-identificatie/0.1",
        "Accept": "application/geo+json",
    })
    herhaling = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    sessie.mount("https://", HTTPAdapter(max_retries=herhaling))
    return sessie


def haal_bouwwerken(bbox, cache_map: Path | None = None, timeout: int = 120) -> gpd.GeoDataFrame:
    """Panden en overige bouwwerken uit de BGT, op of boven het maaiveld.

    Alles met een negatieve hoogteligging valt af: dat ligt onder de grond, en daarboven
    ligt gewoon maaiveld.
    """
    sleutel = hashlib.sha1(json.dumps([round(getal, 1) for getal in bbox]).encode()).hexdigest()[:16]
    cachepad = cache_map / "bgt" / f"bouwwerken_{sleutel}.geojson" if cache_map else None
    if cachepad is not None and cachepad.exists():
        return _naar_gdf(json.loads(cachepad.read_text(encoding="utf-8")), bbox)

    sessie = _sessie()
    kenmerken: list[dict] = []
    for collectie in COLLECTIES:
        try:
            antwoord = sessie.get(
                f"{BASIS_URL}/{collectie}/items",
                params={
                    "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "bbox-crs": RD_URI, "crs": RD_URI, "limit": MAX_OBJECTEN, "f": "json",
                },
                timeout=timeout,
            )
            antwoord.raise_for_status()
            gevonden = antwoord.json().get("features", [])
        except Exception as fout:
            logger.warning("BGT %s niet op te halen (%s)", collectie, fout)
            continue
        for kenmerk in gevonden:
            kenmerk.setdefault("properties", {})["collectie"] = collectie
        kenmerken.extend(gevonden)

    verzameling = {"type": "FeatureCollection", "features": kenmerken}
    if cachepad is not None and kenmerken:
        cachepad.parent.mkdir(parents=True, exist_ok=True)
        cachepad.write_text(json.dumps(verzameling), encoding="utf-8")
    return _naar_gdf(verzameling, bbox)


def _naar_gdf(verzameling: dict, bbox) -> gpd.GeoDataFrame:
    kenmerken = verzameling.get("features", [])
    if not kenmerken:
        return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], dtype="geometry")}, geometry="geometry", crs=RD)
    objecten = gpd.GeoDataFrame.from_features(kenmerken, crs=RD)
    objecten = objecten[~objecten.geometry.is_empty & objecten.geometry.notna()].copy()
    ongeldig = ~objecten.geometry.is_valid
    if ongeldig.any():
        objecten.loc[ongeldig, "geometry"] = objecten.loc[ongeldig, "geometry"].make_valid()

    hoogteligging = objecten.get("relatieve_hoogteligging")
    if hoogteligging is not None:
        ondergronds = hoogteligging.fillna(0).astype(float) < 0
        if ondergronds.any():
            logger.info(
                "%s ondergrondse bouwwerken overgeslagen, daarboven ligt maaiveld", int(ondergronds.sum())
            )
        objecten = objecten[~ondergronds]
    return gpd.clip(objecten, box(*bbox)).reset_index(drop=True)


def bouwwerkmaskers(bbox, vorm, transform, cache_map: Path | None = None, buffer_m: float = 1.0):
    """Twee maskers: het grondvlak zelf, en datzelfde vlak met buffer om van af te snijden.

    Het gebufferde masker vangt overstekende daken en balkons op en wordt gebruikt om het
    maaiveld af te bakenen. Het onbewerkte grondvlak is wat er als bouwwerk uit gaat, want
    anders zou de bouwwerklaag de buffer meetellen.
    """
    from luchtfoto_objecten.raster import polygonen_naar_masker

    bouwwerken = haal_bouwwerken(bbox, cache_map)
    if bouwwerken.empty:
        leeg = np.zeros(vorm, dtype=bool)
        return leeg, leeg
    logger.info("BGT: %s bouwwerken op of boven maaiveld in dit gebied", len(bouwwerken))
    grondvlak = polygonen_naar_masker(bouwwerken.geometry, vorm, transform)
    ruim = polygonen_naar_masker(bouwwerken.geometry, vorm, transform, buffer_m=buffer_m)
    return grondvlak, ruim
