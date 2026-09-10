from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from skimage.measure import label
from skimage.morphology import closing, disk, opening, remove_small_holes, remove_small_objects

from luchtfoto_objecten.detectie.kenmerken import grijswaarde, hsv_kanalen, lokale_standaarddeviatie
from luchtfoto_objecten.geo_hulp import RD, lege_gdf
from luchtfoto_objecten.instellingen import VerhardingParameters
from luchtfoto_objecten.raster import gelabeld_naar_polygonen, meters_naar_pixels, pixeloppervlak

logger = logging.getLogger(__name__)

KOLOMMEN = ["klasse", "oppervlakte_m2", "helderheid", "verzadiging", "textuur", "betrouwbaarheid", "bron"]


def bepaal_verhardingsmasker(
    rgb: np.ndarray,
    transform: Affine,
    params: VerhardingParameters,
    uitsluitmasker: np.ndarray | None = None,
) -> np.ndarray:
    """Vlak, onbegroeid maaiveld: gesloten en open verharding, erven en pleinen.

    De drempels zijn geijkt op de geregistreerde verhardingsvlakken in Centrum. Bruine
    klinkers halen een hogere verzadiging dan asfalt, vandaar de ruime marge. Panden en
    water komen uit de BGT en gaan via het uitsluitmasker het beeld uit, die hoeven we
    dus niet op kleur te onderscheiden.
    """
    _, verzadiging, waarde = hsv_kanalen(rgb)
    textuur = lokale_standaarddeviatie(grijswaarde(rgb), meters_naar_pixels(1.0, transform))

    masker = (
        (verzadiging <= params.max_verzadiging)
        & (waarde >= params.min_helderheid)
        & (waarde <= params.max_helderheid)
        & (textuur <= params.max_textuur)
    )
    if uitsluitmasker is not None:
        masker &= ~uitsluitmasker

    min_pixels = max(8, int(params.min_oppervlakte_m2 / pixeloppervlak(transform)))
    masker = opening(masker, disk(1))
    masker = remove_small_objects(masker, max_size=min_pixels)
    masker = remove_small_holes(masker, max_size=min_pixels)
    return closing(masker, disk(2))


def detecteer_verharding(
    rgb: np.ndarray,
    transform: Affine,
    params: VerhardingParameters,
    uitsluitmasker: np.ndarray | None = None,
    masker: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    if masker is None:
        masker = bepaal_verhardingsmasker(rgb, transform, params, uitsluitmasker)
    if not masker.any():
        logger.info("Geen verharding gevonden")
        return lege_gdf(KOLOMMEN)

    _, verzadiging, waarde = hsv_kanalen(rgb)
    textuur = lokale_standaarddeviatie(grijswaarde(rgb), meters_naar_pixels(1.0, transform))
    componenten = label(masker)

    records = []
    for component_label, geometrie in gelabeld_naar_polygonen(
        componenten, transform, params.min_oppervlakte_m2, params.vereenvoudiging_m
    ):
        component = componenten == component_label
        verzadiging_gemiddeld = float(np.mean(verzadiging[component]))
        textuur_gemiddeld = float(np.median(textuur[component]))
        betrouwbaarheid = 0.5 * np.clip(1 - verzadiging_gemiddeld / max(params.max_verzadiging, 1e-6), 0, 1)
        betrouwbaarheid += 0.5 * np.clip(1 - textuur_gemiddeld / max(params.max_textuur, 1e-6), 0, 1)
        records.append({
            "klasse": "verharding",
            "oppervlakte_m2": round(float(geometrie.area), 2),
            "helderheid": round(float(np.mean(waarde[component])), 3),
            "verzadiging": round(verzadiging_gemiddeld, 3),
            "textuur": round(textuur_gemiddeld, 4),
            "betrouwbaarheid": round(float(betrouwbaarheid), 3),
            "bron": "luchtfoto",
            "geometry": geometrie,
        })

    if not records:
        return lege_gdf(KOLOMMEN)
    logger.info("Verharding gedetecteerd: %s vlakken", len(records))
    return gpd.GeoDataFrame(records, geometry="geometry", crs=RD)
