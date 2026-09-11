"""Bomen, groenvlakken en verharding uit één opname.

Welke opname dat is, kies je: infrarood of kleur. Infrarood staat standaard aan, want NDVI
scheidt vegetatie van verharding veel scherper dan een kleurindex. Gemeten op het
Vliegenbos vangt NDVI boven 0,00 ruim 84% van het geregistreerde begroeide terrein terwijl
maar 7% van de wegdelen meekomt; excess green op kleur bleef daar op enkele procenten
precisie steken.

De drempel wordt niet geraden maar op de registratie geijkt, zodat elke opname en elke
jaargang op zijn eigen werkpunt staat in plaats van op een vast getal dat meedrijft met de
kleurzweem van dat beeld.

Daarna valt het maaiveld in drie klassen uiteen:

    boom          begroeid, en met de vorm van een kroon
    groenvlak     begroeid, maar uitgestrekt of langgerekt
    verharding    het maaiveld dat niet begroeid is

Het onderscheid tussen boom en groenvlak zit in de vorm, niet in de kleur, want een kroon
en een gazon zijn allebei even groen. Een kroon is rond, compact en hooguit een meter of
negen breed; een berm is smal maar eindeloos lang, en een plantsoen is simpelweg te groot.
Die drie grenzen staan onder `groenstructuur` in config/parameters.yaml.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from luchtfoto_objecten.detectie.kenmerken import exces_groen
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.kalibratie import ijk_drempel, referentiemaskers
from luchtfoto_objecten.modeldetectie import _kenmerken_van, naar_laag, opschonen

logger = logging.getLogger(__name__)

BEELDSOORTEN = ("infrarood", "ortho")


def haal_beeld(gebied, instellingen: Instellingen, soort: str | None = None):
    """De opname waarop geclassificeerd wordt, plus de naam van de soort.

    Valt infrarood weg, bijvoorbeeld omdat PDOK het voor dit gebied niet heeft, dan zakken
    we terug naar kleur en melden dat, zodat je het niet ongemerkt op een ander beeld doet.
    """
    from luchtfoto_objecten.infrarood import haal_infrarood
    from luchtfoto_objecten.modeldetectie import haal_beelden

    soort = soort or instellingen.luchtfoto.beeld
    if soort not in BEELDSOORTEN:
        raise SystemExit(f"Onbekend beeld '{soort}'. Kies uit: {', '.join(BEELDSOORTEN)}")

    if soort == "infrarood":
        uitsnede = haal_infrarood(gebied, instellingen)
        if uitsnede is not None:
            return uitsnede, "infrarood"
        logger.warning("Geen infrarood beschikbaar, we classificeren op de kleuropname")

    hoofd, groenbeeld, _, _ = haal_beelden(gebied, instellingen)
    return groenbeeld, "ortho"


def vegetatiegetal(uitsnede, soort: str) -> np.ndarray:
    """NDVI op infrarood, excess green op kleur."""
    if soort == "infrarood":
        from luchtfoto_objecten.infrarood import ndvi

        return ndvi(uitsnede)
    return exces_groen(uitsnede.afbeelding)


def vormkenmerken(vlakken: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compactheid en breedte per vlak, de twee maten die kroon van veld scheiden."""
    tabel = vlakken.copy().reset_index(drop=True)
    oppervlak = tabel.geometry.area
    omtrek = tabel.geometry.length.replace(0, np.nan)
    tabel["oppervlakte_m2"] = oppervlak.round(2)
    # Een cirkel haalt 1,0; hoe langgerekter of grilliger, hoe lager.
    tabel["compactheid"] = (4 * np.pi * oppervlak / (omtrek**2)).fillna(0).round(3)
    # Hydraulische breedte: bij een langgerekte strook ongeveer de breedte ervan.
    tabel["breedte_m"] = (4 * oppervlak / omtrek).fillna(0).round(2)
    return tabel


def is_boom(tabel: gpd.GeoDataFrame, instellingen: Instellingen) -> pd.Series:
    params = instellingen.groenstructuur
    return (
        (tabel["breedte_m"] <= params.boom_max_breedte_m)
        & (tabel["compactheid"] >= params.boom_min_compactheid)
        & (tabel["oppervlakte_m2"] <= params.boom_max_oppervlakte_m2)
    )


def classificeer(
    gebied, instellingen: Instellingen | None = None, soort: str | None = None
) -> tuple[dict[str, gpd.GeoDataFrame], pd.DataFrame, object, str]:
    """Geeft de drie klassen als lagen, een overzichtstabel, de gebruikte uitsnede en de soort."""
    from luchtfoto_objecten.modeldetectie import haal_beelden

    instellingen = instellingen or Instellingen.laden()
    _, _, registratie, analysevlak = haal_beelden(gebied, instellingen)
    uitsnede, gebruikt = haal_beeld(gebied, instellingen, soort)
    kenmerken = _kenmerken_van(uitsnede, registratie, analysevlak, instellingen)

    getal = vegetatiegetal(uitsnede, gebruikt)
    groen_ref, verhard_ref = referentiemaskers(registratie, kenmerken.vorm, uitsnede.transform, kenmerken.maaiveld)
    keuze = ijk_drempel(getal, groen_ref, verhard_ref)
    ondergrens = (
        instellingen.groen.ndvi_ondergrens if gebruikt == "infrarood" else instellingen.groen.exg_ondergrens
    )
    if keuze.bruikbaar and np.isfinite(keuze.drempel):
        drempel = max(keuze.drempel, ondergrens)
        logger.info(
            "Vegetatiedrempel op %s geijkt: %.4f (IoU %.3f, precisie %.3f, recall %.3f, AUC %.3f)%s",
            gebruikt, keuze.drempel, keuze.iou, keuze.precisie, keuze.recall, keuze.scheidend_vermogen,
            "" if drempel == keuze.drempel else f", opgetrokken naar de ondergrens {ondergrens:.4f}",
        )
    else:
        drempel = ondergrens
        logger.warning("Te weinig referentie om te ijken, we gebruiken de ondergrens %.4f", drempel)

    begroeid = (getal > drempel) & kenmerken.maaiveld
    verhard = kenmerken.maaiveld & ~begroeid

    groenvlakken = naar_laag(
        opschonen(begroeid, uitsnede.transform, instellingen.groen.min_oppervlakte_m2),
        uitsnede.transform, "begroeid", gebruikt,
        instellingen.groen.min_oppervlakte_m2, instellingen.groen.vereenvoudiging_m, analysevlak,
    )
    verharding = naar_laag(
        opschonen(verhard, uitsnede.transform, instellingen.verharding.min_oppervlakte_m2),
        uitsnede.transform, "verharding", gebruikt,
        instellingen.verharding.min_oppervlakte_m2, instellingen.verharding.vereenvoudiging_m, analysevlak,
    )

    if groenvlakken.empty:
        bomen = groenvlakken
    else:
        groenvlakken = vormkenmerken(groenvlakken)
        boomvlak = is_boom(groenvlakken, instellingen)
        bomen = groenvlakken[boomvlak].copy()
        bomen["klasse"] = "boom"
        groenvlakken = groenvlakken[~boomvlak].copy()
        groenvlakken["klasse"] = "groenvlak"
        # De stam zit onder het midden van de kroon, dus het zwaartepunt is de plek.
        bomen["middelpunt_x"] = bomen.geometry.centroid.x.round(2)
        bomen["middelpunt_y"] = bomen.geometry.centroid.y.round(2)

    lagen = {
        "classificatie_boom": bomen,
        "classificatie_groenvlak": groenvlakken,
        "classificatie_verharding": verharding,
    }
    overzicht = pd.DataFrame([
        {
            "klasse": naam.replace("classificatie_", ""),
            "beeld": uitsnede.laag,
            "soort": gebruikt,
            "drempel": round(float(drempel), 4),
            "vlakken": int(len(laag)),
            "oppervlakte_m2": round(float(laag.geometry.area.sum()), 1) if not laag.empty else 0.0,
            "aandeel_analysevlak": (
                round(float(laag.geometry.area.sum()) / analysevlak.area, 3)
                if not laag.empty and analysevlak.area else 0.0
            ),
        }
        for naam, laag in lagen.items()
    ])
    return lagen, overzicht, uitsnede, gebruikt
