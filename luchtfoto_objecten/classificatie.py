"""Objecten classificeren uit open bronnen alleen: luchtfoto en hoogtemodel.

Hier wordt niets gespiegeld aan een gemeentelijke registratie. Wat eruit komt is wat de
open bronnen laten zien, en dat is precies wat je nodig hebt als je die registratie later
wilt controleren: een oordeel dat niet al van tevoren naar het antwoord is toegerekend.

Twee bronnen dragen het:

    infrarood   NDVI scheidt begroeid van onbegroeid; band 1 is het nabij-infrarood
    AHN         objecthoogte, het verschil tussen bovenkant en maaiveld

Hoogte is de beslissende maat. Van bovenaf lijken gazon, heestervak en haag op elkaar:
alle drie groen, alle drie plat. Ze verschillen in hoogte. En een kale boom in het voorjaar
heeft geen groensignaal maar wel hoogte, want lidar meet de takken gewoon; daarmee is een
kroon herkenbaar op een opname waar geen blad aan zit.

De klassen:

    gras                 begroeid, tot een halve meter
    heesters             begroeid, tot anderhalve meter
    haag                 begroeid, smal en langgerekt, tot boven kniehoogte
    boomkroon            hoog en grillig van vorm
    dichte_begroeiing    begroeid en hoger dan een heester, maar geen boom of haag
    verharding           onbegroeid en vlak
    bouwwerk             hoog en glad van bovenaf

Zonder AHN vervalt het naar drie klassen, begroeid, verharding en onbekend hoog, en dat
wordt in de uitvoer gemeld. Alle grenzen staan onder `classificatie` in
config/parameters.yaml, want ze hangen af van wat je beheert: een lage haag in de ene
gemeente is een hoge bodembedekker in de andere.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from luchtfoto_objecten.detectie.kenmerken import exces_groen
from luchtfoto_objecten.geo_hulp import RD, lege_gdf
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.raster import masker_naar_polygonen, polygonen_naar_masker

logger = logging.getLogger(__name__)

BEELDSOORTEN = ("infrarood", "ortho")
KLASSEN = ("gras", "heesters", "haag", "boomkroon", "dichte_begroeiing", "verharding", "bouwwerk")


def haal_beeld(gebied, instellingen: Instellingen, soort: str | None = None):
    """De opname waarop geclassificeerd wordt, plus welke soort het werd."""
    from luchtfoto_objecten.infrarood import haal_infrarood
    from luchtfoto_objecten.wmts import LuchtfotoWMTS

    soort = soort or instellingen.luchtfoto.beeld
    if soort not in BEELDSOORTEN:
        raise SystemExit(f"Onbekend beeld '{soort}'. Kies uit: {', '.join(BEELDSOORTEN)}")

    if soort == "infrarood":
        uitsnede = haal_infrarood(gebied, instellingen)
        if uitsnede is not None:
            return uitsnede, "infrarood"
        logger.warning("Geen infrarood beschikbaar, we classificeren op de kleuropname")

    wmts = LuchtfotoWMTS(
        laag=instellingen.luchtfoto.laag, zoom=instellingen.luchtfoto.zoom,
        cache_map=instellingen.cache_map, max_werkers=instellingen.luchtfoto.max_werkers,
    )
    return wmts.haal_uitsnede(gebied.bbox), "ortho"


def vegetatiegetal(uitsnede, soort: str) -> np.ndarray:
    """NDVI op infrarood, excess green op kleur."""
    if soort == "infrarood":
        from luchtfoto_objecten.infrarood import ndvi

        return ndvi(uitsnede)
    return exces_groen(uitsnede.afbeelding)


def maaiveldmasker(uitsnede, objecthoogte, instellingen: Instellingen) -> np.ndarray:
    """Alles behalve wat hoog en glad is, dus zonder daken.

    Zonder registratie is er geen pandenkaart, maar die is ook niet nodig: een dak is
    simpelweg hoog en van bovenaf vlak. Dat haalt het AHN er zelf uit.
    """
    if objecthoogte is None:
        return np.ones(uitsnede.afbeelding.shape[:2], dtype=bool)
    from luchtfoto_objecten.hoogte import hoogteruwheid

    params = instellingen.classificatie
    ruw = hoogteruwheid(objecthoogte, uitsnede.transform)
    bouwwerk = (objecthoogte >= params.bouwwerk_min_hoogte_m) & (ruw <= params.dak_max_ruwheid_m)
    return ~bouwwerk


def deel_in(
    getal: np.ndarray, objecthoogte: np.ndarray | None, ruwheid: np.ndarray | None,
    maaiveld: np.ndarray, drempel: float, instellingen: Instellingen,
) -> dict[str, np.ndarray]:
    """Per klasse een masker, op hoogte en groensignaal."""
    params = instellingen.classificatie
    begroeid = getal > drempel

    if objecthoogte is None:
        return {
            "gras": begroeid & maaiveld,
            "verharding": ~begroeid & maaiveld,
        }

    laag = objecthoogte < params.gras_max_hoogte_m
    heesterhoogte = (objecthoogte >= params.gras_max_hoogte_m) & (objecthoogte < params.heester_max_hoogte_m)
    middenhoog = (objecthoogte >= params.heester_max_hoogte_m) & (objecthoogte < params.boom_min_hoogte_m)
    hoog = objecthoogte >= params.boom_min_hoogte_m

    # Een kroon is hoog en grillig; een dak is even hoog maar glad. Blad is mooi meegenomen
    # maar niet vereist, want op de voorjaarsopname is de kroon kaal.
    grillig = ruwheid > params.kroon_min_ruwheid_m
    bouwwerk = hoog & ~grillig & ~begroeid
    boomkroon = hoog & (grillig | begroeid) & ~bouwwerk

    return {
        "gras": begroeid & laag & maaiveld,
        "heesters": begroeid & heesterhoogte & maaiveld,
        "dichte_begroeiing": begroeid & middenhoog & maaiveld,
        "boomkroon": boomkroon & maaiveld,
        "verharding": ~begroeid & laag & maaiveld,
        "bouwwerk": bouwwerk,
    }


def _gemiddelde_per_vlak(geometrieen, vlak: np.ndarray, transform) -> pd.Series:
    """Gemiddelde van een raster binnen elk vlak, bijvoorbeeld de zekerheid."""
    from rasterio.features import rasterize
    from scipy import ndimage

    labels = rasterize(
        ((geo, index + 1) for index, geo in enumerate(geometrieen)),
        out_shape=vlak.shape, transform=transform, fill=0, dtype=np.int32,
    )
    waarden = ndimage.mean(vlak, labels=labels, index=np.arange(1, len(geometrieen) + 1))
    return pd.Series(np.nan_to_num(waarden), index=geometrieen.index)


def _vormmaten(vlakken: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    oppervlak = vlakken.geometry.area
    omtrek = vlakken.geometry.length.replace(0, np.nan)
    vlakken["oppervlakte_m2"] = oppervlak.round(2)
    vlakken["compactheid"] = (4 * np.pi * oppervlak / (omtrek**2)).fillna(0).round(3)
    vlakken["breedte_m"] = (4 * oppervlak / omtrek).fillna(0).round(2)
    vlakken["lengte_m"] = (omtrek / 2).round(1)
    return vlakken


def _splits_hagen(vlakken: gpd.GeoDataFrame, instellingen: Instellingen):
    """Een haag is smal, lang en recht; een heestervak is een vlek.

    Dat onderscheid zit in de vorm en niet in de hoogte, want ze overlappen in hoogte.
    """
    params = instellingen.classificatie
    if vlakken.empty:
        return vlakken, vlakken
    haag = (
        (vlakken["breedte_m"] <= params.haag_max_breedte_m)
        & (vlakken["lengte_m"] >= params.haag_min_lengte_m)
        & (vlakken["compactheid"] <= params.haag_max_compactheid)
    )
    return vlakken[haag].copy(), vlakken[~haag].copy()


def _naar_vlakken(masker, transform, klasse: str, min_opp: float, vereenvoudiging: float, grens) -> gpd.GeoDataFrame:
    from luchtfoto_objecten.modeldetectie import opschonen

    geometrieen = masker_naar_polygonen(
        opschonen(masker, transform, min_opp), transform, min_opp, vereenvoudiging
    )
    if not geometrieen:
        return lege_gdf(["klasse", "oppervlakte_m2"])
    vlakken = gpd.GeoDataFrame({"klasse": [klasse] * len(geometrieen)}, geometry=geometrieen, crs=RD)
    if grens is not None:
        vlakken = gpd.clip(vlakken, grens)
        vlakken = vlakken[~vlakken.geometry.is_empty & vlakken.geometry.notna()]
        vlakken = vlakken[vlakken.geometry.area >= min_opp]
    return _vormmaten(vlakken.reset_index(drop=True))


def classificeer(
    gebied, instellingen: Instellingen | None = None, soort: str | None = None,
    grens: BaseGeometry | None = None, gebruik_hoogte: bool = True, jaargangen: int = 1,
) -> tuple[dict[str, gpd.GeoDataFrame], pd.DataFrame, object, dict, dict]:
    """Classificeert het gebied op beeld en hoogte, zonder registratie erbij.

    Geeft de lagen per klasse, een overzichtstabel, de gebruikte uitsnede, een verslagje van
    welke bronnen erin zaten, en de invoerrasters zelf, zodat je kunt narekenen waar een
    klasse vandaan komt.
    """
    from luchtfoto_objecten.hoogte import haal_objecthoogte, hoogteruwheid

    instellingen = instellingen or Instellingen.laden()
    params = instellingen.classificatie
    uitsnede, gebruikt = haal_beeld(gebied, instellingen, soort)
    vorm = uitsnede.afbeelding.shape[:2]

    objecthoogte = None
    if gebruik_hoogte:
        objecthoogte = haal_objecthoogte(gebied.bbox, vorm, uitsnede.transform, instellingen.cache_map)
    ruwheid = hoogteruwheid(objecthoogte, uitsnede.transform) if objecthoogte is not None else None

    getal = vegetatiegetal(uitsnede, gebruikt)
    drempel = params.ndvi_drempel if gebruikt == "infrarood" else instellingen.groen.exg_ondergrens
    maaiveld = maaiveldmasker(uitsnede, objecthoogte, instellingen)
    if grens is not None:
        maaiveld &= polygonen_naar_masker([grens], vorm, uitsnede.transform)

    # Meerdere jaargangen samen geven een steviger oordeel dan één opname: wat maar in één
    # jaar groen is, is meestal een auto, een schaduw of een natte plek.
    zekerheid = None
    jaarverslag: list[dict] = []
    maskers_per_jaargang: dict = {}
    if jaargangen > 1 and gebruikt == "infrarood":
        from luchtfoto_objecten.meerjarig_beeld import consensus

        uitkomst = consensus(
            gebied, instellingen, vorm, uitsnede.transform, drempel, jaargangen, binnen=maaiveld
        )
        if uitkomst is not None:
            tellers, zekerheid, jaarverslag, maskers_per_jaargang = uitkomst
            # Meerderheid van de jaargangen, zodat één afwijkend jaar de uitspraak niet kantelt.
            nodig = max(1, (len(jaarverslag) + 1) // 2)
            getal = np.where(tellers >= nodig, max(drempel + 0.01, 1.0), drempel - 1.0).astype(np.float32)
            logger.info(
                "Consensus over %s jaargangen, een pixel telt als begroeid vanaf %s jaar",
                len(jaarverslag), nodig,
            )

    maskers = deel_in(getal, objecthoogte, ruwheid, maaiveld, drempel, instellingen)

    lagen: dict[str, gpd.GeoDataFrame] = {}
    lagen_per_jaargang: dict[str, gpd.GeoDataFrame] = {}
    for klasse, masker in maskers.items():
        min_opp = params.min_oppervlakte_m2 if klasse != "boomkroon" else params.kroon_min_oppervlakte_m2
        vlakken = _naar_vlakken(masker, uitsnede.transform, klasse, min_opp, params.vereenvoudiging_m, grens)
        if klasse == "heesters":
            hagen, heesters = _splits_hagen(vlakken, instellingen)
            if not hagen.empty:
                hagen["klasse"] = "haag"
                lagen["classificatie_haag"] = hagen
            vlakken = heesters
        if zekerheid is not None and not vlakken.empty:
            vlakken["zekerheid"] = _gemiddelde_per_vlak(vlakken.geometry, zekerheid, uitsnede.transform).round(2)
        if klasse == "boomkroon" and not vlakken.empty:
            # Het zwaartepunt van de kroon is de plek van de boom.
            vlakken["middelpunt_x"] = vlakken.geometry.centroid.x.round(2)
            vlakken["middelpunt_y"] = vlakken.geometry.centroid.y.round(2)
        lagen[f"classificatie_{klasse}"] = vlakken

    lagen = voeg_totalen_toe(lagen)
    bronnen = {
        "beeld": uitsnede.laag,
        "beeldsoort": gebruikt,
        "vegetatiedrempel": round(float(drempel), 4),
        "hoogtebron": "AHN dsm_05m/dtm_05m" if objecthoogte is not None else "geen",
        "jaargangen": len(jaarverslag) if jaarverslag else 1,
        "jaarverslag": jaarverslag,
        "klassen": "volledig" if objecthoogte is not None else "beperkt, zonder hoogte",
    }
    overzicht = pd.DataFrame([
        {
            "klasse": naam.replace("classificatie_", ""),
            "vlakken": int(len(laag)),
            "oppervlakte_m2": round(float(laag.geometry.area.sum()), 1) if not laag.empty else 0.0,
            "mediane_breedte_m": round(float(laag["breedte_m"].median()), 2) if not laag.empty else 0.0,
        }
        for naam, laag in sorted(lagen.items())
    ])
    if maskers_per_jaargang:
        bronnen_jaar = {}
        for laagnaam, masker in maskers_per_jaargang.items():
            for klasse, deelmasker in deel_in(
                np.where(masker, drempel + 1.0, drempel - 1.0).astype(np.float32),
                objecthoogte, ruwheid, maaiveld, drempel, instellingen,
            ).items():
                vlakken = _naar_vlakken(
                    deelmasker, uitsnede.transform, klasse,
                    params.min_oppervlakte_m2 if klasse != "boomkroon" else params.kroon_min_oppervlakte_m2,
                    params.vereenvoudiging_m, grens,
                )
                if not vlakken.empty:
                    bronnen_jaar[f"{laagnaam}_{klasse}"] = vlakken
        lagen_per_jaargang.update(bronnen_jaar)

    invoer = {"vegetatiegetal": getal}
    if zekerheid is not None:
        invoer["zekerheid_jaargangen"] = zekerheid
    if objecthoogte is not None:
        invoer["objecthoogte"] = objecthoogte
        invoer["hoogteruwheid"] = ruwheid
    bronnen["lagen_per_jaargang"] = lagen_per_jaargang
    return lagen, overzicht, uitsnede, bronnen, invoer


GROENKLASSEN = ("gras", "heesters", "haag", "dichte_begroeiing")


def voeg_totalen_toe(lagen: dict[str, gpd.GeoDataFrame]) -> dict[str, gpd.GeoDataFrame]:
    """Eén laag groen en één laag verharding, om snel op oppervlak te kunnen kijken.

    Het groentotaal is vegetatie op het maaiveld, dus zonder de boomkronen: die hangen
    erboven en zouden het maaiveld dubbel tellen.
    """
    groen = [lagen[f"classificatie_{k}"] for k in GROENKLASSEN if not lagen.get(f"classificatie_{k}", gpd.GeoDataFrame()).empty]
    if groen:
        totaal = gpd.GeoDataFrame(pd.concat(groen, ignore_index=True), geometry="geometry", crs=groen[0].crs)
        totaal["klasse"] = "groen_totaal"
        lagen["classificatie_groen_totaal"] = totaal

    verharding = lagen.get("classificatie_verharding")
    if verharding is not None and not verharding.empty:
        totaal = verharding.copy()
        totaal["klasse"] = "verharding_totaal"
        lagen["classificatie_verharding_totaal"] = totaal
    return lagen


def varianten() -> list[tuple[str, str, dict]]:
    """Tien instellingen om de klassengrenzen mee af te tasten.

    De eerste is de standaard. Daarna verschuift telkens één grens, zodat je kunt zien wat
    die grens doet zonder dat er iets anders meebeweegt.
    """
    return [
        ("01_standaard", "NDVI 0.05, gras tot 0.5 m, heester tot 1.5 m, boom vanaf 3 m", {}),
        ("02_ndvi_ruim_002", "NDVI-drempel 0.02, meer vegetatie", {"ndvi_drempel": 0.02}),
        ("03_ndvi_streng_010", "NDVI-drempel 0.10, minder vegetatie", {"ndvi_drempel": 0.10}),
        ("04_gras_tot_030", "gras tot 0.3 m, meer telt als heester", {"gras_max_hoogte_m": 0.3}),
        ("05_gras_tot_080", "gras tot 0.8 m, minder telt als heester", {"gras_max_hoogte_m": 0.8}),
        ("06_heester_tot_100", "heester tot 1.0 m", {"heester_max_hoogte_m": 1.0}),
        ("07_heester_tot_250", "heester tot 2.5 m", {"heester_max_hoogte_m": 2.5}),
        ("08_boom_vanaf_200", "boom vanaf 2 m, meer kronen", {"boom_min_hoogte_m": 2.0}),
        ("09_boom_vanaf_500", "boom vanaf 5 m, alleen forse bomen", {"boom_min_hoogte_m": 5.0}),
        ("10_kroon_ruwer_100", "kroon pas bij ruwheid 1.0 m, strenger tegen daken", {"kroon_min_ruwheid_m": 1.0}),
    ]


def classificeer_object(geometrie: BaseGeometry, instellingen: Instellingen | None = None, **opties) -> pd.DataFrame:
    """Classificeert één object: welke klassen liggen erin en voor hoeveel.

    Handig om een enkel geregistreerd vlak na te lopen zonder een heel gebied te draaien.
    """
    from luchtfoto_objecten.gebieden import gebied_uit_bbox

    marge = 5.0
    xmin, ymin, xmax, ymax = geometrie.bounds
    gebied = gebied_uit_bbox((xmin - marge, ymin - marge, xmax + marge, ymax + marge), naam="object")
    lagen, _, _, bronnen, _ = classificeer(gebied, instellingen, grens=geometrie, **opties)

    oppervlak = geometrie.area
    rijen = []
    for naam, laag in sorted(lagen.items()):
        if laag.empty:
            continue
        gedeeld = laag.geometry.union_all().intersection(geometrie).area
        if gedeeld <= 0:
            continue
        rijen.append({
            "klasse": naam.replace("classificatie_", ""),
            "oppervlakte_m2": round(gedeeld, 2),
            "aandeel": round(gedeeld / oppervlak, 3) if oppervlak else 0.0,
        })
    tabel = pd.DataFrame(rijen).sort_values("aandeel", ascending=False).reset_index(drop=True)
    for sleutel, waarde in bronnen.items():
        tabel[sleutel] = waarde
    return tabel


def sweep_klassen(gebied, instellingen: Instellingen | None = None, grens=None, gebruik_hoogte: bool = True):
    """Draait de tien instellingen en geeft per variant de lagen en een overzicht.

    De lagen krijgen de variantnaam mee, zodat ze in één GeoPackage naast elkaar kunnen
    staan en je ze in QGIS één voor één kunt aanzetten.
    """
    from dataclasses import replace

    instellingen = instellingen or Instellingen.laden()
    alle_lagen: dict[str, gpd.GeoDataFrame] = {}
    regels = []
    for naam, omschrijving, afwijking in varianten():
        variant = Instellingen.laden()
        variant.luchtfoto = instellingen.luchtfoto
        variant.classificatie = replace(instellingen.classificatie, **afwijking)
        lagen, overzicht, _, _, _ = classificeer(
            gebied, variant, grens=grens, gebruik_hoogte=gebruik_hoogte
        )
        for laagnaam, laag in lagen.items():
            if not laag.empty:
                alle_lagen[f"{naam}_{laagnaam.replace('classificatie_', '')}"] = laag
        overzicht = overzicht.copy()
        overzicht.insert(0, "variant", naam)
        overzicht.insert(1, "instelling", omschrijving)
        regels.append(overzicht)
        logger.info("Variant %s: %s", naam, omschrijving)
    return alle_lagen, pd.concat(regels, ignore_index=True)
