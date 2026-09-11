"""Drempels ijken op de registratie in plaats van ze te raden.

Otsu zoekt de diepste dal in het histogram van het hele beeld. Dat dal ligt zelden waar de
grens tussen gras en bestrating ligt: op de bladopname van Noord kiest Otsu 0,062 terwijl
0,034 de beste scheiding geeft, en dat verschil is precies het gras dat wordt gemist. Een
vaste waarde uit config werkt net zo min, want elke jaargang heeft zijn eigen kleurzweem.

Wat wel werkt is de gemeentelijke registratie als ijkpunt gebruiken. Die ligt er toch al,
en hoeft niet perfect te zijn om een drempel mee te kiezen: we vragen hem alleen waar de
scheiding ongeveer ligt, niet waar precies elke rand loopt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

MONSTERPIXELS = 200_000


@dataclass(frozen=True)
class Drempelkeuze:
    drempel: float
    iou: float
    precisie: float
    recall: float
    scheidend_vermogen: float
    aantal_positief: int
    aantal_negatief: int

    @property
    def bruikbaar(self) -> bool:
        return self.aantal_positief > 0 and self.aantal_negatief > 0


def _monster(waarden: np.ndarray, masker: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gekozen = waarden[masker]
    gekozen = gekozen[np.isfinite(gekozen)]
    if gekozen.size > MONSTERPIXELS:
        gekozen = rng.choice(gekozen, MONSTERPIXELS, replace=False)
    return gekozen


def scheidend_vermogen(positief: np.ndarray, negatief: np.ndarray) -> float:
    """Kans dat een willekeurige groenpixel hoger scoort dan een willekeurige verhardingspixel.

    Oftewel de AUC. Onafhankelijk van de drempel, dus bruikbaar om beelden te vergelijken
    die elk hun eigen kleurzweem hebben.
    """
    if positief.size == 0 or negatief.size == 0:
        return 0.5
    alles = np.concatenate([positief, negatief])
    rangorde = np.argsort(np.argsort(alles)) + 1
    rangsom = rangorde[: positief.size].sum()
    return float((rangsom - positief.size * (positief.size + 1) / 2) / (positief.size * negatief.size))


def ijk_drempel(
    waarden: np.ndarray,
    positief_masker: np.ndarray,
    negatief_masker: np.ndarray,
    zaad: int = 0,
) -> Drempelkeuze:
    """Zoekt de drempel waarbij de overlap met de registratie het grootst is."""
    rng = np.random.default_rng(zaad)
    positief = _monster(waarden, positief_masker, rng)
    negatief = _monster(waarden, negatief_masker, rng)
    if positief.size < 100 or negatief.size < 100:
        return Drempelkeuze(float("nan"), 0.0, 0.0, 0.0, 0.5, positief.size, negatief.size)

    positief_gesorteerd = np.sort(positief)
    negatief_gesorteerd = np.sort(negatief)
    kandidaten = np.percentile(np.concatenate([positief, negatief]), np.linspace(1, 99, 99))

    # IoU = tp / (aantal positieven + fp), want tp + fn is per definitie het aantal positieven.
    boven_positief = positief.size - np.searchsorted(positief_gesorteerd, kandidaten, side="right")
    boven_negatief = negatief.size - np.searchsorted(negatief_gesorteerd, kandidaten, side="right")
    iou = boven_positief / np.maximum(positief.size + boven_negatief, 1)

    beste = int(np.argmax(iou))
    tp, fp = boven_positief[beste], boven_negatief[beste]
    return Drempelkeuze(
        drempel=float(kandidaten[beste]),
        iou=float(iou[beste]),
        precisie=float(tp / max(tp + fp, 1)),
        recall=float(tp / max(positief.size, 1)),
        scheidend_vermogen=scheidend_vermogen(positief, negatief),
        aantal_positief=int(positief.size),
        aantal_negatief=int(negatief.size),
    )


def referentiemaskers(
    registratie: dict,
    vorm: tuple[int, int],
    transform,
    maaiveld: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Zet de registratie om in twee ijkvlakken: zeker groen en zeker verhard.

    Voor verharding nemen we alleen wegdelen. Onbegroeide terreindelen zijn in de BGT
    grotendeels erven, en daar staan schuurtjes, hagen en bosschages op die het ijkpunt
    zouden vervuilen.
    """
    from luchtfoto_objecten.raster import polygonen_naar_masker

    def masker_van(sleutels: list[str]) -> np.ndarray:
        masker = np.zeros(vorm, dtype=bool)
        for sleutel in sleutels:
            laag = registratie.get(sleutel)
            if laag is not None and not laag.empty:
                masker |= polygonen_naar_masker(laag.geometry, vorm, transform)
        return masker

    groen = masker_van(["begroeideterreindelen", "groenobjecten"])
    verhard = masker_van(["wegdelen"]) & ~groen
    if maaiveld is not None:
        groen &= maaiveld
        verhard &= maaiveld
    return groen, verhard
