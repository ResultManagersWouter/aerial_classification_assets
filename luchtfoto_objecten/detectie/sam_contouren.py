from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine

from luchtfoto_objecten.geo_hulp import RD
from luchtfoto_objecten.raster import masker_naar_polygonen

logger = logging.getLogger(__name__)

INSTALLATIE_HINT = (
    "Voor contourverfijning met SAM: pip install -r requirements-sam.txt en download een checkpoint, "
    "bijvoorbeeld sam_vit_b_01ec64.pth van https://github.com/facebookresearch/segment-anything"
)


class SamContourVerfijner:
    """Optioneel: scherpt de contouren van gevonden objecten aan met Segment Anything.

    SAM heeft geen trainingsdata nodig. Het model krijgt de zwaartepunten van de klassieke
    detectie als aanwijzing en levert daar een nauwkeuriger masker bij terug.
    """

    def __init__(self, checkpoint: str, modeltype: str = "vit_b", apparaat: str | None = None) -> None:
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as fout:
            raise ImportError(f"{fout}. {INSTALLATIE_HINT}") from fout

        if apparaat is None:
            if torch.cuda.is_available():
                apparaat = "cuda"
            elif torch.backends.mps.is_available():
                apparaat = "mps"
            else:
                apparaat = "cpu"
        logger.info("SAM laden (%s) op %s", modeltype, apparaat)
        model = sam_model_registry[modeltype](checkpoint=checkpoint)
        model.to(apparaat)
        self.voorspeller = SamPredictor(model)
        self.apparaat = apparaat

    def verfijn(
        self,
        rgb: np.ndarray,
        transform: Affine,
        objecten: gpd.GeoDataFrame,
        max_objecten: int = 200,
    ) -> gpd.GeoDataFrame:
        if objecten.empty:
            return objecten
        if len(objecten) > max_objecten:
            logger.warning(
                "Meer objecten (%s) dan het maximum van %s, alleen de grootste worden verfijnd",
                len(objecten), max_objecten,
            )
            objecten = objecten.assign(_oppervlak=objecten.geometry.area).nlargest(max_objecten, "_oppervlak")
            objecten = objecten.drop(columns="_oppervlak")

        self.voorspeller.set_image(rgb)
        omgekeerd = ~transform
        verfijnd = []
        for _, rij in objecten.iterrows():
            punt = rij.geometry.representative_point()
            kolom, regel = omgekeerd * (punt.x, punt.y)
            maskers, scores, _ = self.voorspeller.predict(
                point_coords=np.array([[kolom, regel]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            beste = int(np.argmax(scores))
            polygonen = masker_naar_polygonen(maskers[beste].astype(bool), transform, min_oppervlakte_m2=1.0)
            if not polygonen:
                continue
            grootste = max(polygonen, key=lambda geo: geo.area)
            nieuw = rij.to_dict()
            nieuw["geometry"] = grootste
            nieuw["oppervlakte_m2"] = round(float(grootste.area), 2)
            nieuw["sam_score"] = round(float(scores[beste]), 3)
            verfijnd.append(nieuw)

        if not verfijnd:
            return objecten
        logger.info("SAM heeft %s contouren verfijnd", len(verfijnd))
        return gpd.GeoDataFrame(verfijnd, geometry="geometry", crs=RD)
