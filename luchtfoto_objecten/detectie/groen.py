from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, disk, opening, remove_small_holes, remove_small_objects

from luchtfoto_objecten.detectie.kenmerken import exces_groen, grijswaarde, lokale_standaarddeviatie
from luchtfoto_objecten.geo_hulp import RD, lege_gdf
from luchtfoto_objecten.instellingen import GroenParameters
from luchtfoto_objecten.raster import gelabeld_naar_polygonen, meters_naar_pixels, pixeloppervlak

logger = logging.getLogger(__name__)

KOLOMMEN = ["klasse", "oppervlakte_m2", "exg_gemiddeld", "textuur", "betrouwbaarheid", "bron"]


def bepaal_groenmasker(
    rgb: np.ndarray,
    transform: Affine,
    params: GroenParameters,
    uitsluitmasker: np.ndarray | None = None,
    drempel: float | None = None,
) -> np.ndarray:
    """Vegetatie volgens de excess green index.

    Werkt alleen op een opname met blad. Op de voorjaarsopname van de 8cm-ortho zijn gras
    en bestrating spectraal vrijwel gelijk, zie bladstand.bepaal_groenbron.

    Geef een drempel mee als die op de registratie geijkt is. Zonder drempel valt hij terug
    op Otsu, en die kiest stelselmatig te hoog: op de bladopname van Noord 0,062 tegen 0,034
    geijkt, wat het verschil is tussen een gemist en een gevonden grasveld.
    """
    exg = exces_groen(rgb)
    if drempel is None:
        drempel = _groendrempel(exg, params.exg_ondergrens)
        logger.info("Groendrempel via Otsu: %.4f", drempel)
    else:
        logger.info("Groendrempel geijkt op de registratie: %.4f", drempel)

    masker = exg > drempel
    if uitsluitmasker is not None:
        masker &= ~uitsluitmasker

    min_pixels = max(4, int(params.min_oppervlakte_m2 / pixeloppervlak(transform)))
    masker = opening(masker, disk(1))
    masker = remove_small_objects(masker, max_size=min_pixels)
    masker = remove_small_holes(masker, max_size=min_pixels)
    return closing(masker, disk(1))


def detecteer_groen(
    rgb: np.ndarray,
    transform: Affine,
    params: GroenParameters,
    uitsluitmasker: np.ndarray | None = None,
    masker: np.ndarray | None = None,
    drempel: float | None = None,
) -> gpd.GeoDataFrame:
    if masker is None:
        masker = bepaal_groenmasker(rgb, transform, params, uitsluitmasker, drempel)
    if not masker.any():
        logger.info("Geen vegetatie gevonden boven de drempel")
        return lege_gdf(KOLOMMEN)

    exg = exces_groen(rgb)
    if drempel is None:
        drempel = _groendrempel(exg, params.exg_ondergrens)
    textuur = lokale_standaarddeviatie(grijswaarde(rgb), meters_naar_pixels(params.textuur_venster_m, transform))
    componenten = label(masker)

    records = []
    for component_label, geometrie in gelabeld_naar_polygonen(
        componenten, transform, params.min_oppervlakte_m2, params.vereenvoudiging_m
    ):
        component = componenten == component_label
        exg_gemiddeld = float(np.mean(exg[component]))
        records.append({
            "klasse": "groen",
            "oppervlakte_m2": round(float(geometrie.area), 2),
            "exg_gemiddeld": round(exg_gemiddeld, 4),
            "textuur": round(float(np.median(textuur[component])), 4),
            "betrouwbaarheid": round(float(np.clip(0.40 + (exg_gemiddeld - drempel) / 0.12 * 0.60, 0.0, 1.0)), 3),
            "bron": "luchtfoto",
            "geometry": geometrie,
        })

    if not records:
        return lege_gdf(KOLOMMEN)
    logger.info("Groen gedetecteerd: %s vlakken", len(records))
    return gpd.GeoDataFrame(records, geometry="geometry", crs=RD)


def _groendrempel(exg: np.ndarray, ondergrens: float) -> float:
    try:
        otsu = float(threshold_otsu(exg[np.isfinite(exg)]))
    except ValueError:
        otsu = ondergrens
    return max(otsu, ondergrens)
