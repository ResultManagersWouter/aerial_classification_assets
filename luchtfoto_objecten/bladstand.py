from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from luchtfoto_objecten.detectie.kenmerken import exces_groen
from luchtfoto_objecten.wmts import LuchtfotoWMTS, haal_beschikbare_lagen, jaarlagen_nieuwste_eerst

logger = logging.getLogger(__name__)

# De 8cm-ortho wordt in het vroege voorjaar gevlogen, met kale bomen zodat het maaiveld
# zichtbaar is. Dat is prettig voor verharding, maar vegetatie is dan nauwelijks van
# bestrating te onderscheiden. Voor groen zoeken we de nieuwste jaargang met blad.
MONSTERZOOM = 13
MONSTERGROOTTE_M = 150.0
MAXIMALE_ZOOM_25CM = 14
MAX_KANDIDATEN = 8


@dataclass
class Bladstandkeuze:
    laag: str
    zoom: int
    groenaandeel: float
    gemeten: dict[str, float] = field(default_factory=dict)

    @property
    def heeft_blad(self) -> bool:
        return self.groenaandeel > 0


def meet_groenaandeel(rgb: np.ndarray, drempel: float = 0.05) -> float:
    """Aandeel pixels met een duidelijk groensignaal, ruwe maat voor de bladstand."""
    return float((exces_groen(rgb) > drempel).mean())


def bepaal_groenbron(
    bbox: tuple[float, float, float, float],
    hoofdlaag: str,
    zoom: int,
    cache_map: Path | None,
    voorkeur: str = "auto",
    minimaal_aandeel: float = 0.10,
) -> Bladstandkeuze:
    """Kiest de luchtfotolaag waarop vegetatie het best zichtbaar is.

    Met voorkeur 'auto' bemonsteren we een klein vlak uit het midden van het gebied en
    nemen we de eerste laag die genoeg blad laat zien. Een expliciete laagnaam wordt
    zonder meting overgenomen, 'gelijk' houdt de hoofdlaag aan.
    """
    if voorkeur == "gelijk":
        return Bladstandkeuze(laag=hoofdlaag, zoom=zoom, groenaandeel=-1.0)
    if voorkeur != "auto":
        return Bladstandkeuze(laag=voorkeur, zoom=_passende_zoom(voorkeur, zoom), groenaandeel=-1.0)

    monster_bbox = _monstervlak(bbox)
    gemeten: dict[str, float] = {}
    for laag in _kandidaten(hoofdlaag, cache_map):
        try:
            uitsnede = LuchtfotoWMTS(
                laag=laag, zoom=MONSTERZOOM, cache_map=cache_map, max_werkers=2
            ).haal_uitsnede(monster_bbox, toon_voortgang=False)
        except Exception as fout:
            logger.warning("Laag %s kon niet worden bemonsterd: %s", laag, fout)
            continue
        aandeel = meet_groenaandeel(uitsnede.afbeelding)
        gemeten[laag] = round(aandeel, 4)
        logger.info("Bladstand %s: %.1f%% groene pixels", laag, aandeel * 100)
        if aandeel >= minimaal_aandeel:
            return Bladstandkeuze(
                laag=laag, zoom=_passende_zoom(laag, zoom), groenaandeel=aandeel, gemeten=gemeten
            )

    if not gemeten:
        return Bladstandkeuze(laag=hoofdlaag, zoom=zoom, groenaandeel=-1.0)
    beste = max(gemeten, key=gemeten.get)
    logger.warning(
        "Geen enkele laag haalt %.0f%% groen, we gebruiken %s met %.1f%%",
        minimaal_aandeel * 100, beste, gemeten[beste] * 100,
    )
    return Bladstandkeuze(laag=beste, zoom=_passende_zoom(beste, zoom), groenaandeel=gemeten[beste], gemeten=gemeten)


def _kandidaten(hoofdlaag: str, cache_map: Path | None) -> list[str]:
    """De hoofdlaag eerst, daarna alle jaargangen van nieuw naar oud.

    Alleen RGB. De bladstand wordt met excess green gemeten en dat getal zegt niets op een
    infraroodopname, waar vegetatie juist rood is; die lagen horen hier dus niet tussen.
    """
    lagen = {naam: titel for naam, titel in haal_beschikbare_lagen(cache_map).items() if not naam.endswith("IR")}
    jaarlagen = jaarlagen_nieuwste_eerst(lagen)
    volgorde = [hoofdlaag] + [laag for laag in jaarlagen if laag != hoofdlaag]
    return volgorde[:MAX_KANDIDATEN]


def _passende_zoom(laag: str, zoom: int) -> int:
    """Bij een 25cm-laag heeft doorschalen naar 5cm geen zin, dat levert alleen meer tegels."""
    if "ortho25" in laag:
        return min(zoom, MAXIMALE_ZOOM_25CM)
    return zoom


def _monstervlak(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    midden_x, midden_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    halve_breedte = min(MONSTERGROOTTE_M, xmax - xmin) / 2
    halve_hoogte = min(MONSTERGROOTTE_M, ymax - ymin) / 2
    return (midden_x - halve_breedte, midden_y - halve_hoogte, midden_x + halve_breedte, midden_y + halve_hoogte)
