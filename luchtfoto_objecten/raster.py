from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

RD = "EPSG:28992"


def schrijf_geotiff(pad: Path, afbeelding: np.ndarray, transform: Affine, crs: str = RD) -> Path:
    pad.parent.mkdir(parents=True, exist_ok=True)
    hoogte, breedte = afbeelding.shape[:2]
    banden = afbeelding.shape[2] if afbeelding.ndim == 3 else 1
    with rasterio.open(
        pad, "w", driver="GTiff", height=hoogte, width=breedte, count=banden,
        dtype=afbeelding.dtype, crs=crs, transform=transform, compress="deflate", tiled=True,
    ) as bestand:
        if banden == 1:
            bestand.write(afbeelding, 1)
        else:
            for index in range(banden):
                bestand.write(afbeelding[:, :, index], index + 1)
    return pad


def schrijf_raster_in_geopackage(
    pad: Path, naam: str, afbeelding: np.ndarray, transform: Affine, crs: str = RD
) -> Path:
    """Zet het bronbeeld als rasterlaag in een bestaande GeoPackage.

    Zo zit de foto waarop de classificatie gebaseerd is in hetzelfde bestand als de
    vlakken, en hoef je in QGIS niet apart een achtergrond te zoeken. De tegels gaan als
    JPEG het bestand in, want dit is fotomateriaal en anders wordt het bestand veel groter.
    """
    hoogte, breedte = afbeelding.shape[:2]
    banden = afbeelding.shape[2] if afbeelding.ndim == 3 else 1
    with rasterio.open(
        pad, "w", driver="GPKG", height=hoogte, width=breedte, count=banden, dtype=afbeelding.dtype,
        crs=crs, transform=transform, RASTER_TABLE=naam, APPEND_SUBDATASET="YES",
        TILE_FORMAT="JPEG", QUALITY="85",
    ) as bestand:
        if banden == 1:
            bestand.write(afbeelding, 1)
        else:
            for index in range(banden):
                bestand.write(afbeelding[:, :, index], index + 1)
    return pad


def lees_geotiff(pad: Path) -> tuple[np.ndarray, Affine, str]:
    with rasterio.open(pad) as bestand:
        afbeelding = np.transpose(bestand.read(), (1, 2, 0))
        return afbeelding, bestand.transform, str(bestand.crs)


def pixeloppervlak(transform: Affine) -> float:
    return abs(transform.a * transform.e)


def meters_naar_pixels(meters: float, transform: Affine) -> int:
    return max(1, int(round(meters / abs(transform.a))))


def masker_naar_polygonen(
    masker: np.ndarray,
    transform: Affine,
    min_oppervlakte_m2: float = 0.0,
    vereenvoudiging_m: float = 0.0,
) -> list[BaseGeometry]:
    geometrieen = []
    for vorm, waarde in shapes(masker.astype(np.uint8), mask=masker, transform=transform):
        if not waarde:
            continue
        geometrie = shape(vorm)
        if not geometrie.is_valid:
            geometrie = geometrie.buffer(0)
        if geometrie.is_empty or geometrie.area < min_oppervlakte_m2:
            continue
        if vereenvoudiging_m > 0:
            geometrie = geometrie.simplify(vereenvoudiging_m, preserve_topology=True)
        if not geometrie.is_empty and geometrie.area >= min_oppervlakte_m2:
            geometrieen.append(geometrie)
    return geometrieen


def gelabeld_naar_polygonen(
    labels: np.ndarray,
    transform: Affine,
    min_oppervlakte_m2: float = 0.0,
    vereenvoudiging_m: float = 0.0,
) -> list[tuple[int, BaseGeometry]]:
    """Zet een gelabeld raster om in polygonen, één per label."""
    resultaat = []
    for vorm, waarde in shapes(labels.astype(np.int32), mask=labels > 0, transform=transform):
        geometrie = shape(vorm)
        if not geometrie.is_valid:
            geometrie = geometrie.buffer(0)
        if geometrie.is_empty or geometrie.area < min_oppervlakte_m2:
            continue
        if vereenvoudiging_m > 0:
            geometrie = geometrie.simplify(vereenvoudiging_m, preserve_topology=True)
        if not geometrie.is_empty:
            resultaat.append((int(waarde), geometrie))
    return resultaat


def polygonen_naar_masker(
    geometrieen, vorm: tuple[int, int], transform: Affine, buffer_m: float = 0.0,
    all_touched: bool = True,
) -> np.ndarray:
    lijst = [geo.buffer(buffer_m) if buffer_m else geo for geo in geometrieen if geo is not None and not geo.is_empty]
    if not lijst:
        return np.zeros(vorm, dtype=bool)
    return rasterize(
        ((geo, 1) for geo in lijst), out_shape=vorm, transform=transform, fill=0, dtype=np.uint8,
        all_touched=all_touched,
    ).astype(bool)
