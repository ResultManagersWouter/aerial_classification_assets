from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from luchtfoto_objecten.geo_hulp import maak_schrijfbaar

logger = logging.getLogger(__name__)


def schrijf_geopackage(lagen: dict[str, gpd.GeoDataFrame], pad: Path) -> Path:
    pad.parent.mkdir(parents=True, exist_ok=True)
    if pad.exists():
        pad.unlink()
    geschreven = 0
    for naam, laag in lagen.items():
        if laag is None or laag.empty:
            logger.info("Laag %s is leeg en wordt overgeslagen", naam)
            continue
        maak_schrijfbaar(laag).to_file(pad, layer=naam, driver="GPKG")
        geschreven += 1
    logger.info("GeoPackage weggeschreven met %s lagen: %s", geschreven, pad)
    return pad


def schrijf_geojson(lagen: dict[str, gpd.GeoDataFrame], map_pad: Path) -> list[Path]:
    map_pad.mkdir(parents=True, exist_ok=True)
    paden = []
    for naam, laag in lagen.items():
        if laag is None or laag.empty:
            continue
        bestand = map_pad / f"{naam}.geojson"
        maak_schrijfbaar(laag).to_file(bestand, driver="GeoJSON")
        paden.append(bestand)
    logger.info("GeoJSON weggeschreven: %s bestanden in %s", len(paden), map_pad)
    return paden


def schrijf_samenvatting(samenvatting: dict, per_thema: pd.DataFrame, map_pad: Path) -> tuple[Path, Path]:
    map_pad.mkdir(parents=True, exist_ok=True)
    json_pad = map_pad / "samenvatting.json"
    csv_pad = map_pad / "samenvatting_per_thema.csv"
    json_pad.write_text(json.dumps(samenvatting, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    per_thema.to_csv(csv_pad, index=False, sep=";", decimal=",")
    return json_pad, csv_pad
