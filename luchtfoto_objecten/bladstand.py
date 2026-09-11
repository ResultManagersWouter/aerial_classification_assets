from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from luchtfoto_objecten.detectie.kenmerken import exces_groen
from luchtfoto_objecten.kalibratie import ijk_drempel, referentiemaskers
from luchtfoto_objecten.wmts import LuchtfotoWMTS, haal_beschikbare_lagen, jaarlagen_nieuwste_eerst

logger = logging.getLogger(__name__)

# De 8cm-ortho wordt in het vroege voorjaar gevlogen, met kale bomen zodat het maaiveld
# zichtbaar is. Dat is prettig voor verharding, maar vegetatie is dan nauwelijks van
# bestrating te onderscheiden. Voor groen zoeken we daarom een andere opname.
#
# De keuze gaat in twee stappen. Eerst een goedkope zeef op bladstand, die kale winters en
# lege tegels wegneemt. Daarna scoren de overgebleven lagen op wat werkelijk telt: hoe goed
# vegetatie er van verharding te scheiden is, gemeten tegen de registratie. Alleen kijken
# naar "hoeveel groen zie ik" kiest namelijk het verkeerde beeld. De 25cm-zomeropname van
# Amsterdam heeft een groene waas over het hele beeld, tot in de wegen toe, en haalt daarmee
# 80% groene pixels tegen 36% voor de bladopname op 8cm, terwijl die laatste veel beter
# scheidt.
MONSTERZOOM = 13
MONSTERGROOTTE_M = 150.0
SCOREGROOTTE_M = 200.0
MAXIMALE_ZOOM_25CM = 14
MAX_KANDIDATEN = 8
RESOLUTIEMARGE = 0.03


@dataclass
class Bladstandkeuze:
    laag: str
    zoom: int
    groenaandeel: float
    gemeten: dict[str, float] = field(default_factory=dict)
    scheiding: dict[str, float] = field(default_factory=dict)

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
    referentie: dict | None = None,
) -> Bladstandkeuze:
    """Kiest de luchtfotolaag waarop vegetatie het best van verharding te scheiden is.

    Met een registratie erbij wordt er gescoord op die scheiding. Zonder registratie valt de
    keuze terug op de eerste laag met genoeg blad, wat ruwer is maar zonder gemeentedata werkt.
    """
    if voorkeur == "gelijk":
        return Bladstandkeuze(laag=hoofdlaag, zoom=zoom, groenaandeel=-1.0)
    if voorkeur != "auto":
        return Bladstandkeuze(laag=voorkeur, zoom=_passende_zoom(voorkeur, zoom), groenaandeel=-1.0)

    gemeten = _meet_bladstand(bbox, hoofdlaag, cache_map)
    if not gemeten:
        return Bladstandkeuze(laag=hoofdlaag, zoom=zoom, groenaandeel=-1.0)

    met_blad = [laag for laag, aandeel in gemeten.items() if aandeel >= minimaal_aandeel]
    if not met_blad:
        beste = max(gemeten, key=gemeten.get)
        logger.warning(
            "Geen enkele laag haalt %.0f%% groen, we gebruiken %s met %.1f%%",
            minimaal_aandeel * 100, beste, gemeten[beste] * 100,
        )
        return Bladstandkeuze(beste, _passende_zoom(beste, zoom), gemeten[beste], gemeten)

    if referentie is None:
        laag = met_blad[0]
        logger.info("Zonder registratie kiezen we de nieuwste laag met blad: %s", laag)
        return Bladstandkeuze(laag, _passende_zoom(laag, zoom), gemeten[laag], gemeten)

    scheiding = _scoor_scheiding(bbox, met_blad, zoom, cache_map, referentie)
    if not scheiding:
        laag = met_blad[0]
        return Bladstandkeuze(laag, _passende_zoom(laag, zoom), gemeten[laag], gemeten)

    laag = _kies_met_resolutievoorkeur(scheiding, zoom)
    logger.info(
        "Groenbron %s gekozen op scheidend vermogen (IoU %.3f tegen de registratie)",
        laag, scheiding[laag],
    )
    return Bladstandkeuze(
        laag=laag, zoom=_passende_zoom(laag, zoom), groenaandeel=gemeten[laag],
        gemeten=gemeten, scheiding={k: round(v, 4) for k, v in scheiding.items()},
    )


def _meet_bladstand(bbox, hoofdlaag: str, cache_map: Path | None) -> dict[str, float]:
    """Goedkope zeef: een klein vlak op laag zoomniveau, puur om kale opnamen weg te nemen."""
    monster_bbox = _vlak_rond_midden(bbox, MONSTERGROOTTE_M)
    gemeten: dict[str, float] = {}
    for laag in _kandidaten(hoofdlaag, cache_map):
        try:
            uitsnede = LuchtfotoWMTS(
                laag=laag, zoom=MONSTERZOOM, cache_map=cache_map, max_werkers=2
            ).haal_uitsnede(monster_bbox, toon_voortgang=False)
        except Exception as fout:
            logger.warning("Laag %s kon niet worden bemonsterd: %s", laag, fout)
            continue
        if uitsnede.afbeelding.min() > 250:
            logger.info("Bladstand %s: geen dekking op deze plek", laag)
            continue
        aandeel = meet_groenaandeel(uitsnede.afbeelding)
        gemeten[laag] = round(aandeel, 4)
        logger.info("Bladstand %s: %.1f%% groene pixels", laag, aandeel * 100)
    return gemeten


def _scoor_scheiding(bbox, lagen: list[str], zoom: int, cache_map, referentie: dict) -> dict[str, float]:
    """Scoort elke laag op de overlap met de registratie bij de best mogelijke drempel."""
    score_bbox = _vlak_rond_midden(bbox, SCOREGROOTTE_M)
    scores: dict[str, float] = {}
    for laag in lagen:
        eigen_zoom = _passende_zoom(laag, zoom)
        try:
            uitsnede = LuchtfotoWMTS(
                laag=laag, zoom=eigen_zoom, cache_map=cache_map, max_werkers=4
            ).haal_uitsnede(score_bbox, toon_voortgang=False)
        except Exception as fout:
            logger.warning("Laag %s kon niet worden gescoord: %s", laag, fout)
            continue
        vorm = uitsnede.afbeelding.shape[:2]
        groen_ref, verhard_ref = referentiemaskers(referentie, vorm, uitsnede.transform)
        keuze = ijk_drempel(exces_groen(uitsnede.afbeelding), groen_ref, verhard_ref)
        if not keuze.bruikbaar or np.isnan(keuze.drempel):
            logger.info("Laag %s: te weinig referentievlakken om te scoren", laag)
            continue
        scores[laag] = keuze.iou
        logger.info(
            "Scheiding %s: IoU %.3f bij drempel %.4f (AUC %.3f)",
            laag, keuze.iou, keuze.drempel, keuze.scheidend_vermogen,
        )
    return scores


def _kies_met_resolutievoorkeur(scores: dict[str, float], zoom: int) -> str:
    """Bij een gelijk spel wint de fijnste opname, want die geeft bruikbaarder contouren."""
    beste_score = max(scores.values())
    dicht_bij = [laag for laag, score in scores.items() if beste_score - score <= RESOLUTIEMARGE]
    return min(dicht_bij, key=lambda laag: (_resolutierang(laag), -scores[laag]))


def _resolutierang(laag: str) -> int:
    return 1 if "ortho25" in laag else 0


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


def _vlak_rond_midden(bbox, grootte_m: float):
    xmin, ymin, xmax, ymax = bbox
    midden_x, midden_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    halve_breedte = min(grootte_m, xmax - xmin) / 2
    halve_hoogte = min(grootte_m, ymax - ymin) / 2
    return (midden_x - halve_breedte, midden_y - halve_hoogte, midden_x + halve_breedte, midden_y + halve_hoogte)
