"""Objecthoogte uit het AHN, de maat die een foto niet kan geven.

Van bovenaf lijken een gazon, een heestervak en een haag op elkaar: alle drie groen, alle
drie plat. Ze verschillen vooral in hoogte, en dat is precies wat een nadiropname niet
meet. Het AHN wel.

Het hoogtemodel komt van de WCS van PDOK, als twee lagen: `dsm_05m` is de bovenkant van
alles wat er staat, `dtm_05m` het maaiveld eronder. Het verschil is de objecthoogte, het
genormaliseerde oppervlaktemodel. Beide staan al in RD op 50 centimeter, dus er hoeft niets
omgeprojecteerd te worden, alleen hergeschaald naar het raster van de foto.

Twee dingen die dit oplost en een foto niet kan.

Een kale boom in het voorjaar heeft geen groensignaal, maar wel hoogte: het AHN is lidar en
meet gewoon de takken. Daarmee is een boomkroon herkenbaar op een opname waar geen blad
aan zit, en dat was precies waar de beeldclassificatie op vastliep.

En een dak is net zo hoog als een kroon, maar veel vlakker. De ruwheid van de hoogte, de
spreiding binnen een meter, scheidt die twee: een kroon is grillig, een dak is glad.

Let op de peildatum. Het AHN wordt niet elk jaar ingewonnen, dus in een pas opgeleverde
wijk kan het achterlopen op de luchtfoto. Waar het AHN niets heeft, valt de classificatie
terug op het beeld alleen en wordt dat als zodanig gemeld.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from luchtfoto_objecten.detectie.kenmerken import lokale_standaarddeviatie
from luchtfoto_objecten.raster import meters_naar_pixels

logger = logging.getLogger(__name__)

WCS_URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
BOVENKANT = "dsm_05m"
MAAIVELD = "dtm_05m"
MAX_PIXELS = 40_000_000  # ruim een vierkante kilometer op 50 cm


def _sessie() -> requests.Session:
    sessie = requests.Session()
    sessie.headers.update({"User-Agent": "luchtfoto-objecten-identificatie/0.1"})
    herhaling = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    sessie.mount("https://", HTTPAdapter(max_retries=herhaling))
    return sessie


def _haal_laag(laag: str, bbox, cache_map: Path | None, timeout: int = 180):
    """Eén AHN-laag als array plus transform, met een bestandscache."""
    sleutel = hashlib.sha1(f"{laag}|{[round(g, 1) for g in bbox]}".encode()).hexdigest()[:16]
    cachepad = cache_map / "ahn" / f"{laag}_{sleutel}.tif" if cache_map else None
    if cachepad is not None and cachepad.exists():
        with rasterio.open(cachepad) as bestand:
            return bestand.read(1).astype(np.float32), bestand.transform, bestand.nodata

    antwoord = _sessie().get(
        WCS_URL,
        params={
            "service": "WCS", "version": "2.0.1", "request": "GetCoverage", "coverageId": laag,
            "subset": [f"x({bbox[0]},{bbox[2]})", f"y({bbox[1]},{bbox[3]})"], "format": "image/tiff",
        },
        timeout=timeout,
    )
    antwoord.raise_for_status()
    if cachepad is not None:
        cachepad.parent.mkdir(parents=True, exist_ok=True)
        cachepad.write_bytes(antwoord.content)
    with rasterio.open(io.BytesIO(antwoord.content)) as bestand:
        return bestand.read(1).astype(np.float32), bestand.transform, bestand.nodata


def haal_objecthoogte(bbox, doelvorm, doeltransform, cache_map: Path | None = None) -> np.ndarray | None:
    """Objecthoogte in meters, hergeschaald naar het raster van de foto.

    Geeft None terug als het AHN niets levert voor dit gebied, bijvoorbeeld buiten Nederland
    of bij een storing. De classificatie valt dan terug op het beeld alleen.
    """
    breedte, hoogte_m = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if (breedte / 0.5) * (hoogte_m / 0.5) > MAX_PIXELS:
        logger.warning("Gebied te groot voor het AHN in één keer (%.0f bij %.0f m)", breedte, hoogte_m)
        return None

    try:
        bovenkant, transform, nodata = _haal_laag(BOVENKANT, bbox, cache_map)
        maaiveld, _, nodata_maaiveld = _haal_laag(MAAIVELD, bbox, cache_map)
    except Exception as fout:
        logger.warning("AHN niet op te halen (%s), we classificeren zonder hoogte", fout)
        return None

    for vlak, leeg in ((bovenkant, nodata), (maaiveld, nodata_maaiveld)):
        if leeg is not None:
            vlak[vlak == leeg] = np.nan
        vlak[vlak > 1e30] = np.nan

    objecthoogte = bovenkant - maaiveld
    gemeten = np.isfinite(objecthoogte)
    if not gemeten.any():
        logger.warning("Het AHN heeft geen dekking voor dit gebied")
        return None
    logger.info(
        "AHN opgehaald: mediaan %.2f m, p99 %.1f m, %.0f%% van het gebied gemeten",
        float(np.nanmedian(objecthoogte)), float(np.nanpercentile(objecthoogte, 99)), 100 * gemeten.mean(),
    )
    return _naar_raster(np.nan_to_num(objecthoogte, nan=0.0), transform, doelvorm, doeltransform)


def _naar_raster(bron: np.ndarray, brontransform, doelvorm, doeltransform) -> np.ndarray:
    """Van 50 cm AHN naar het fijnere raster van de luchtfoto."""
    doel = np.zeros(doelvorm, dtype=np.float32)
    reproject(
        source=bron, destination=doel,
        src_transform=brontransform, src_crs="EPSG:28992",
        dst_transform=doeltransform, dst_crs="EPSG:28992",
        resampling=Resampling.bilinear,
    )
    return doel


def hoogteruwheid(objecthoogte: np.ndarray, transform, venster_m: float = 1.5) -> np.ndarray:
    """Spreiding van de hoogte binnen een venster: hoog bij een kroon, laag bij een dak."""
    return lokale_standaarddeviatie(objecthoogte, meters_naar_pixels(venster_m, transform))
