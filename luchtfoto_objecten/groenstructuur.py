"""Scheidt boomkronen van groen op het maaiveld.

De groendetectie is vooral goed in bomen. Dat is logisch: een kroon is het felste groen
op de foto. Maar daardoor komt een grasveld met bomen erop eruit als een verzameling
kronen, en niet als het grasveld dat het is.

Dit model gebruikt de twee opnamen voor waar ze elk goed in zijn. De zomeropname laat
zien waar blad hangt. De 8cm-ortho is in het vroege voorjaar gevlogen met kale bomen, en
laat dus de grond ónder die kroon zien; de verhardingsdetectie daarop vertelt of die
ondergrond hard of zacht is. Gras onder een boomrij is in de zomer onzichtbaar, maar in
het voorjaar gewoon te zien.

Zo ontstaan twee lagen die elkaar mogen overlappen, want dat is de werkelijkheid: de
kroon hangt boven de grond.

    boomkroon    groen op de zomerfoto met de ruwe textuur van blad en takken
    groenstrook  maaiveld dat begroeid is, ook waar een kroon eroverheen hangt

Het groensignaal is leidend. De onverharde ondergrond telt mee als steunbewijs, niet als
doorslaggevend, precies zoals gevraagd: een pixel die groen en glad is telt al als
groenstrook, onverhard maakt die conclusie sterker, en ruwe textuur pleit ertegen. De
gewichten staan in config/parameters.yaml onder `groenstructuur`.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from luchtfoto_objecten.instellingen import GroenstructuurParameters, Instellingen
from luchtfoto_objecten.modeldetectie import Beeldkenmerken, naar_laag, opschonen

logger = logging.getLogger(__name__)


def textuurdrempel(kenmerken: Beeldkenmerken, groen: np.ndarray, params: GroenstructuurParameters) -> float:
    """De grens tussen blad en gras, bepaald binnen het groen zelf.

    Binnen het groen zijn de twee grootste groepen juist kroon en gras, dus hier is Otsu
    wel op zijn plaats, anders dan bij de groendrempel zelf.
    """
    if params.textuurdrempel > 0:
        return params.textuurdrempel
    monster = kenmerken.vlakken["textuur_3m"][groen]
    monster = monster[np.isfinite(monster)]
    if monster.size < 100:
        return float("inf")
    try:
        return float(threshold_otsu(monster))
    except ValueError:
        return float("inf")


def splits(
    kenmerken: Beeldkenmerken, groen: np.ndarray, verhard_eronder: np.ndarray, params: GroenstructuurParameters
) -> tuple[np.ndarray, np.ndarray, float]:
    """Geeft het kroonmasker, het groenstrookmasker en de gebruikte textuurdrempel."""
    drempel = textuurdrempel(kenmerken, groen, params)
    ruw = kenmerken.vlakken["textuur_3m"] > drempel
    onverhard = kenmerken.maaiveld & ~verhard_eronder

    kroon = groen & ruw & kenmerken.maaiveld

    # Stemmen in plaats van harde regels, zodat de afweging verschuifbaar is.
    score = params.gewicht_groen * groen.astype(np.float32)
    score += params.gewicht_onverhard * onverhard.astype(np.float32)
    score -= params.gewicht_ruw * ruw.astype(np.float32)
    strook = (score >= params.drempel) & kenmerken.maaiveld

    logger.info(
        "Textuurdrempel kroon/gras: %.4f, kroon %.0f m2, groenstrook %.0f m2",
        drempel, kroon.sum() * _pixel(kenmerken), strook.sum() * _pixel(kenmerken),
    )
    return kroon, strook, drempel


def _pixel(kenmerken: Beeldkenmerken) -> float:
    return abs(kenmerken.transform.a * kenmerken.transform.e)


def bouw_lagen(
    kenmerken: Beeldkenmerken, groen: np.ndarray, verhard_eronder: np.ndarray,
    analysevlak, instellingen: Instellingen,
) -> tuple[dict[str, gpd.GeoDataFrame], pd.DataFrame]:
    params = instellingen.groenstructuur
    kroon, strook, drempel = splits(kenmerken, groen, verhard_eronder, params)

    lagen: dict[str, gpd.GeoDataFrame] = {}
    overzicht: list[dict] = []
    for naam, masker, min_opp in (
        ("detectie_boomkroon", kroon, params.kroon_min_oppervlakte_m2),
        ("detectie_groenstrook", strook, params.strook_min_oppervlakte_m2),
    ):
        schoon = opschonen(masker, kenmerken.transform, min_opp)
        laag = naar_laag(
            schoon, kenmerken.transform, naam.replace("detectie_", ""), "groenstructuur",
            min_opp, params.vereenvoudiging_m, analysevlak,
        )
        lagen[naam] = laag
        oppervlak = float(laag.geometry.area.sum()) if not laag.empty else 0.0
        overzicht.append({
            "laag": naam,
            "vlakken": int(len(laag)),
            "oppervlakte_m2": round(oppervlak, 1),
            "aandeel_analysevlak": round(oppervlak / analysevlak.area, 3) if analysevlak.area else 0.0,
            "textuurdrempel": round(drempel, 4),
        })
    return lagen, pd.DataFrame(overzicht)


def toets_tegen_register(
    lagen: dict[str, gpd.GeoDataFrame], bomen: gpd.GeoDataFrame, groenobjecten: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Hoeveel geregistreerde bomen vallen in de kroonlaag, en hoeveel geregistreerd groen
    in de groenstrooklaag.

    Dit is geen absolute waarheid - niet elke boom staat geregistreerd en de groenobjecten
    zijn juist wat we controleren - maar het laat wel zien of de twee lagen de goede kant
    op wijzen.
    """
    regels = []
    kroon = lagen.get("detectie_boomkroon")
    if kroon is not None and not kroon.empty and not bomen.empty:
        vlak = kroon.geometry.union_all()
        raak = int(sum(vlak.intersects(punt) for punt in bomen.geometry))
        regels.append({
            "toets": "geregistreerde bomen in de kroonlaag",
            "totaal": int(len(bomen)),
            "geraakt": raak,
            "aandeel": round(raak / len(bomen), 3),
        })

    strook = lagen.get("detectie_groenstrook")
    if strook is not None and not strook.empty and not groenobjecten.empty:
        vlak = strook.geometry.union_all()
        geregistreerd = groenobjecten.geometry.area.sum()
        gedekt = sum(geo.intersection(vlak).area for geo in groenobjecten.geometry)
        regels.append({
            "toets": "geregistreerd groen gedekt door de groenstrooklaag",
            "totaal": round(float(geregistreerd), 1),
            "geraakt": round(float(gedekt), 1),
            "aandeel": round(float(gedekt / geregistreerd), 3) if geregistreerd else 0.0,
        })
    return pd.DataFrame(regels)
