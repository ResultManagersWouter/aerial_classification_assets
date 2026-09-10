from __future__ import annotations

import json
from datetime import date, datetime

import geopandas as gpd
import pandas as pd

RD = "EPSG:28992"


def lege_gdf(kolommen: list[str], crs: str = RD) -> gpd.GeoDataFrame:
    leeg = {kolom: pd.Series(dtype="object") for kolom in kolommen}
    leeg["geometry"] = gpd.GeoSeries([], dtype="geometry")
    return gpd.GeoDataFrame(leeg, geometry="geometry", crs=crs)


def maak_schrijfbaar(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Zet kolommen met lijsten, dicts of datums om naar tekst zodat GeoPackage ze accepteert."""
    kopie = gdf.copy()
    for kolom in kopie.columns:
        if kolom == kopie.geometry.name:
            continue
        waarden = kopie[kolom]
        if waarden.dtype == "object":
            kopie[kolom] = waarden.map(_naar_tekst)
    return kopie


def _naar_tekst(waarde):
    if waarde is None or isinstance(waarde, (str, int, float, bool)):
        return waarde
    if isinstance(waarde, (dict, list, tuple)):
        return json.dumps(waarde, ensure_ascii=False, default=str)
    if isinstance(waarde, (datetime, date)):
        return waarde.isoformat()
    return str(waarde)
