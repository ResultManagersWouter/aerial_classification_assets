"""Vegetatie op de infraroodopname, en daarmee bomen los van echte groenvelden.

PDOK levert de 8cm-ortho ook als infrarood, op een eigen service. De eerste band is het
nabij-infrarood; dat is hier gemeten en niet aangenomen: binnen geregistreerd begroeid
terrein haalt band 1 gemiddeld 155 tegen 112 op wegdelen, terwijl band 2 en 3 daar juist
lager liggen. Daarmee is NDVI = (nir - rood) / (nir + rood) te berekenen.

NDVI is veel scherper dan excess green op RGB. Op het Vliegenbos vangt een drempel van 0,00
ruim 84% van het geregistreerde begroeide terrein, terwijl maar 7% van de wegdelen meekomt.
De regels op RGB haalden daar een precisie van enkele procenten.

Let op wat deze opname wel en niet laat zien. De hele infraroodreeks bij PDOK is in het
vroege voorjaar gevlogen, met kale bomen: in het Vliegenbos, een bos, is de mediane NDVI
negatief. Er is dus geen infrarood mét blad. Dat is hier geen gebrek maar de hele truc,
want door een kale kroon heen zie je de grond eronder:

    hoge NDVI in het voorjaar          gras of beplanting op het maaiveld, ook onder bomen
    groen in de zomer, lage NDVI       kroon boven iets dat zelf niet begroeid is

Zo valt een boomkroon langs de weg, het wolkje groen waar het om begonnen was, aan te
wijzen zonder dat je hem uit textuur hoeft te raden.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.filters import threshold_otsu

from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.wmts import LuchtfotoWMTS, haal_beschikbare_lagen

logger = logging.getLogger(__name__)

NDVI_DREMPEL = 0.0


def infraroodlaag(hoofdlaag: str, cache_map=None) -> str | None:
    """De infraroodtegenhanger van een RGB-laag, als die bestaat."""
    kandidaat = f"{hoofdlaag}IR"
    lagen = haal_beschikbare_lagen(cache_map)
    if kandidaat in lagen:
        return kandidaat
    if "Actueel_orthoHRIR" in lagen:
        return "Actueel_orthoHRIR"
    return None


def haal_infrarood(gebied, instellingen: Instellingen, laag: str | None = None):
    laag = laag or infraroodlaag(instellingen.luchtfoto.laag, instellingen.cache_map)
    if laag is None:
        logger.warning("Geen infraroodlaag beschikbaar")
        return None
    wmts = LuchtfotoWMTS(
        laag=laag, zoom=instellingen.luchtfoto.zoom, cache_map=instellingen.cache_map,
        max_werkers=instellingen.luchtfoto.max_werkers,
    )
    return wmts.haal_uitsnede(gebied.bbox)


def ndvi(uitsnede) -> np.ndarray:
    """Band 1 is het nabij-infrarood, band 2 het rood. Gemeten, zie de moduletekst.

    De tegels komen als JPEG binnen, dus dit is een weergegeven NDVI en geen gekalibreerde
    reflectie. Voor het onderscheid vegetatie tegen verharding is dat ruim genoeg, maar
    vergelijk de absolute waarde niet met NDVI uit een satellietproduct.
    """
    beeld = uitsnede.afbeelding.astype(np.float32)
    nir, rood = beeld[:, :, 0], beeld[:, :, 1]
    return (nir - rood) / np.maximum(nir + rood, 1e-6)


def ndvi_drempel(waarden: np.ndarray, maaiveld: np.ndarray, terugval: float = NDVI_DREMPEL) -> float:
    monster = waarden[maaiveld]
    monster = monster[np.isfinite(monster)]
    if monster.size < 100:
        return terugval
    try:
        return float(threshold_otsu(monster[:: max(1, monster.size // 200_000)]))
    except ValueError:
        return terugval


def splits_kroon_en_veld(
    ndvi_voorjaar: np.ndarray, groen_zomer: np.ndarray, maaiveld: np.ndarray, drempel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Geeft het kroonmasker en het groenveldmasker.

    Een groenveld is maaiveld dat op de voorjaarsopname begroeid is. Een kroon is groen op
    de zomerfoto waar het voorjaar geen begroeiing laat zien: het blad hing er in de zomer
    wel, maar de grond eronder is kaal of verhard.
    """
    begroeid_maaiveld = (ndvi_voorjaar > drempel) & maaiveld
    kroon = groen_zomer & ~begroeid_maaiveld & maaiveld
    return kroon, begroeid_maaiveld
