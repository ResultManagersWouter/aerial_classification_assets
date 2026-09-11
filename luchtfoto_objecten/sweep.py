"""Draait per klasse tien parameterinstellingen en zet ze in aparte GeoPackages.

Bedoeld om er met de luchtfoto ernaast doorheen te lopen en te kiezen. Elke klasse krijgt
een eigen bestand, zodat je er in QGIS één opent en de lagen op volgorde afgaat:

    output/<gebied>/groen.gpkg        tien manieren om vegetatie te vinden
    output/<gebied>/verharding.gpkg   tien manieren om verharding te vinden
    output/<gebied>/bomen.gpkg        tien manieren om kroon van maaiveld te scheiden

De lagen zijn genummerd, zodat ze in QGIS op volgorde staan, en de naam zegt wat er
veranderd is. Naast elk bestand komt een CSV met de parameters en wat ze opleveren.

Alle varianten krijgen dezelfde nabewerking en hetzelfde analysevlak, dus het verschil
dat je ziet komt van de parameter en niet van de opschoning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from luchtfoto_objecten.groenstructuur import splits
from luchtfoto_objecten.instellingen import GroenstructuurParameters, Instellingen
from luchtfoto_objecten.modeldetectie import (
    Beeldkenmerken,
    _kenmerken_van,
    _naar_ander_raster,
    haal_beelden,
    kmeans_groen,
    naar_laag,
    opschonen,
)
from luchtfoto_objecten.raster import schrijf_raster_in_geopackage
from luchtfoto_objecten.uitvoer import schrijf_geopackage

logger = logging.getLogger(__name__)


def _otsu(waarden: np.ndarray, binnen: np.ndarray, terugval: float) -> float:
    monster = waarden[binnen]
    monster = monster[np.isfinite(monster)]
    if monster.size < 100:
        return terugval
    try:
        return float(threshold_otsu(monster))
    except ValueError:
        return terugval


def groenvarianten(k: Beeldkenmerken) -> list[tuple[str, str, np.ndarray]]:
    """Tien manieren om vegetatie aan te wijzen, van streng naar ruim."""
    exg, vari, gli = k.vlakken["exg"], k.vlakken["vari"], k.vlakken["gli"]
    otsu_beeld = _otsu(exg, np.ones(k.vorm, dtype=bool), 0.035)
    otsu_maaiveld = _otsu(exg, k.maaiveld, 0.035)
    return [
        ("01_otsu_heel_beeld", f"huidige regels, drempel {otsu_beeld:.3f}", exg > otsu_beeld),
        ("02_otsu_maaiveld", f"Otsu alleen op maaiveld, drempel {otsu_maaiveld:.3f}", exg > otsu_maaiveld),
        ("03_otsu_begrensd_005", "Otsu maar nooit hoger dan 0.05", exg > min(otsu_maaiveld, 0.05)),
        ("04_exg_vast_0080", "vaste drempel 0.080", exg > 0.080),
        ("05_exg_vast_0050", "vaste drempel 0.050", exg > 0.050),
        ("06_exg_vast_0035", "vaste drempel 0.035, de ondergrens uit config", exg > 0.035),
        ("07_exg_vast_0020", "vaste drempel 0.020, zeer ruim", exg > 0.020),
        ("08_vari_otsu", f"VARI met Otsu op maaiveld", vari > _otsu(vari, k.maaiveld, 0.0)),
        ("09_gli_otsu", "Green Leaf Index met Otsu op maaiveld", gli > _otsu(gli, k.maaiveld, 0.0)),
        ("10_hsv_tint", "groene tint in HSV, verzadiging boven 0.12",
         (k.vlakken["tint"] > 0.15) & (k.vlakken["tint"] < 0.45) & (k.vlakken["verzadiging"] > 0.12)),
    ]


def verhardingsvarianten(k: Beeldkenmerken, instellingen: Instellingen, groen: np.ndarray | None):
    """Tien instellingen voor verharding, rond de drempels uit config."""
    verz, held, tex = k.vlakken["verzadiging"], k.vlakken["helderheid"], k.vlakken["textuur_1m"]
    p = instellingen.verharding

    def regels(max_verz: float, max_tex: float, min_held: float = None, max_held: float = None) -> np.ndarray:
        return (
            (verz <= max_verz)
            & (held >= (p.min_helderheid if min_held is None else min_held))
            & (held <= (p.max_helderheid if max_held is None else max_held))
            & (tex <= max_tex)
        )

    varianten = [
        ("01_huidig", f"verzadiging {p.max_verzadiging}, textuur {p.max_textuur}",
         regels(p.max_verzadiging, p.max_textuur)),
        ("02_verzadiging_ruim_045", "verzadiging tot 0.45", regels(0.45, p.max_textuur)),
        ("03_verzadiging_streng_025", "verzadiging tot 0.25", regels(0.25, p.max_textuur)),
        ("04_textuur_ruim_015", "textuur tot 0.15", regels(p.max_verzadiging, 0.15)),
        ("05_textuur_streng_006", "textuur tot 0.06", regels(p.max_verzadiging, 0.06)),
        ("06_donker_toegestaan", "ook donkere verharding, helderheid vanaf 0.10",
         regels(p.max_verzadiging, p.max_textuur, min_held=0.10)),
        ("07_ruim_totaal", "verzadiging 0.45 en textuur 0.15 samen", regels(0.45, 0.15)),
        ("08_streng_totaal", "verzadiging 0.25 en textuur 0.06 samen", regels(0.25, 0.06)),
    ]
    if groen is not None:
        varianten.append(("09_niet_groen", "alles op het maaiveld dat niet groen is", k.maaiveld & ~groen))
    varianten.append(("10_kmeans", "kleurgroepen die vlak en onverzadigd zijn", _kmeans_verhard(k, instellingen)))
    return varianten


def _kmeans_verhard(k: Beeldkenmerken, instellingen: Instellingen, aantal: int = 6) -> np.ndarray:
    from luchtfoto_objecten.modeldetectie import _kmeans_verharding

    return _kmeans_verharding(k, instellingen, aantal)


def boomvarianten(k: Beeldkenmerken, groen: np.ndarray, verhard_eronder: np.ndarray):
    """Tien afwegingen tussen kroon en maaiveld.

    De kroon hangt aan de textuurdrempel, de groenstrook aan de gewichten. De eerste zes
    varianten lopen daarom de textuurdrempel af, de laatste vier houden die vast op Otsu en
    verschuiven de afweging voor het maaiveld. Zo verschilt elke laag echt van de vorige.
    """
    def maak(**afwijking) -> GroenstructuurParameters:
        params = GroenstructuurParameters()
        for sleutel, waarde in afwijking.items():
            setattr(params, sleutel, waarde)
        return params

    instellingen = [
        ("01_textuur_otsu", "textuurdrempel via Otsu, standaardgewichten", maak()),
        ("02_textuur_004", "textuurdrempel 0.04, veel kroon", maak(textuurdrempel=0.04)),
        ("03_textuur_006", "textuurdrempel 0.06", maak(textuurdrempel=0.06)),
        ("04_textuur_008", "textuurdrempel 0.08", maak(textuurdrempel=0.08)),
        ("05_textuur_010", "textuurdrempel 0.10", maak(textuurdrempel=0.10)),
        ("06_textuur_012", "textuurdrempel 0.12, weinig kroon", maak(textuurdrempel=0.12)),
        ("07_onverhard_uit", "Otsu, onverharde ondergrond telt niet mee", maak(gewicht_onverhard=0.0)),
        ("08_onverhard_zwaar", "Otsu, onverharde ondergrond weegt 1.0", maak(gewicht_onverhard=1.0)),
        ("09_strook_ruim", "Otsu, drempel 0.5 dus ruimere groenstrook", maak(drempel=0.5)),
        ("10_strook_streng", "Otsu, drempel 1.0 dus strengere groenstrook", maak(drempel=1.0)),
    ]
    varianten = []
    for naam, omschrijving, params in instellingen:
        kroon, strook, drempel = splits(k, groen, verhard_eronder, params)
        varianten.append((f"kroon_{naam}", f"{omschrijving} (drempel {drempel:.3f})", kroon))
        varianten.append((f"strook_{naam}", omschrijving, strook))
    return varianten


def _schrijf(
    varianten, k: Beeldkenmerken, analysevlak, klasse: str, min_opp: float, vereenvoudiging: float,
    pad: Path, uitsnede=None,
) -> pd.DataFrame:
    lagen: dict[str, gpd.GeoDataFrame] = {}
    regels = []
    for naam, omschrijving, masker in varianten:
        schoon = opschonen(masker & k.maaiveld, k.transform, min_opp)
        laag = naar_laag(schoon, k.transform, klasse, naam, min_opp, vereenvoudiging, analysevlak)
        lagen[naam] = laag
        oppervlak = float(laag.geometry.area.sum()) if not laag.empty else 0.0
        regels.append({
            "laag": naam,
            "instelling": omschrijving,
            "vlakken": int(len(laag)),
            "oppervlakte_m2": round(oppervlak, 1),
            "aandeel_analysevlak": round(oppervlak / analysevlak.area, 3) if analysevlak.area else 0.0,
        })
        logger.info("%s/%s: %s vlakken, %.0f m2", klasse, naam, len(laag), oppervlak)

    gevuld = {naam: laag for naam, laag in lagen.items() if not laag.empty}
    if gevuld:
        schrijf_geopackage(gevuld, pad)
        if uitsnede is not None:
            schrijf_raster_in_geopackage(pad, f"luchtfoto_{uitsnede.laag}", uitsnede.afbeelding, uitsnede.transform)
    return pd.DataFrame(regels)


def voer_sweep_uit(gebied, instellingen: Instellingen | None, uitvoer_map: Path) -> dict[str, pd.DataFrame]:
    instellingen = instellingen or Instellingen.laden()
    hoofd, groenbeeld, registratie, analysevlak = haal_beelden(gebied, instellingen)
    groen_k = _kenmerken_van(groenbeeld, registratie, analysevlak, instellingen)
    verharding_k = _kenmerken_van(hoofd, registratie, analysevlak, instellingen)
    uitvoer_map.mkdir(parents=True, exist_ok=True)

    groentabel = _schrijf(
        groenvarianten(groen_k), groen_k, analysevlak, "groen",
        instellingen.groen.min_oppervlakte_m2, instellingen.groen.vereenvoudiging_m,
        uitvoer_map / "groen.gpkg", uitsnede=groenbeeld,
    )

    # Het groen dat de verhardingsvarianten als tegenhanger gebruiken, en de verharding die
    # de boomvarianten als ondergrond gebruiken, komen van de standaardinstelling.
    basisgroen = opschonen(
        (groen_k.vlakken["tint"] > 0.15) & (groen_k.vlakken["tint"] < 0.45)
        & (groen_k.vlakken["verzadiging"] > 0.12) & groen_k.maaiveld,
        groen_k.transform, instellingen.groen.min_oppervlakte_m2,
    )
    groen_op_hoofd = _naar_ander_raster(basisgroen, groen_k, verharding_k)

    verhardingstabel = _schrijf(
        verhardingsvarianten(verharding_k, instellingen, groen_op_hoofd), verharding_k, analysevlak,
        "verharding", instellingen.verharding.min_oppervlakte_m2, instellingen.verharding.vereenvoudiging_m,
        uitvoer_map / "verharding.gpkg", uitsnede=hoofd,
    )

    verhard_basis = opschonen(
        (verharding_k.vlakken["verzadiging"] <= instellingen.verharding.max_verzadiging)
        & (verharding_k.vlakken["helderheid"] >= instellingen.verharding.min_helderheid)
        & (verharding_k.vlakken["helderheid"] <= instellingen.verharding.max_helderheid)
        & (verharding_k.vlakken["textuur_1m"] <= instellingen.verharding.max_textuur)
        & verharding_k.maaiveld,
        verharding_k.transform, instellingen.verharding.min_oppervlakte_m2,
    )
    verhard_eronder = _naar_ander_raster(verhard_basis, verharding_k, groen_k)
    if verhard_eronder is None:
        verhard_eronder = np.zeros(groen_k.vorm, dtype=bool)

    boomtabel = _schrijf(
        boomvarianten(groen_k, basisgroen, verhard_eronder), groen_k, analysevlak, "boom",
        instellingen.groenstructuur.kroon_min_oppervlakte_m2, instellingen.groenstructuur.vereenvoudiging_m,
        uitvoer_map / "bomen.gpkg", uitsnede=groenbeeld,
    )

    tabellen = {"groen": groentabel, "verharding": verhardingstabel, "bomen": boomtabel}

    infraroodtabel = _infrarood_sweep(gebied, instellingen, registratie, analysevlak, groen_k, basisgroen, uitvoer_map)
    if infraroodtabel is not None:
        tabellen["infrarood"] = infraroodtabel

    for naam, tabel in tabellen.items():
        tabel.to_csv(uitvoer_map / f"sweep_{naam}.csv", index=False)
    return tabellen


def _infrarood_sweep(gebied, instellingen, registratie, analysevlak, groen_k, basisgroen, uitvoer_map):
    """NDVI-drempels op de voorjaars-infraroodopname, plus de scheiding kroon en groenveld."""
    from luchtfoto_objecten.infrarood import haal_infrarood, ndvi, ndvi_drempel, splits_kroon_en_veld

    infrarood = haal_infrarood(gebied, instellingen)
    if infrarood is None:
        return None

    ir_k = _kenmerken_van(infrarood, registratie, analysevlak, instellingen)
    waarde = ndvi(infrarood)
    otsu = ndvi_drempel(waarde, ir_k.maaiveld)
    groen_op_ir = _naar_ander_raster(basisgroen, groen_k, ir_k)
    if groen_op_ir is None:
        groen_op_ir = np.zeros(ir_k.vorm, dtype=bool)

    varianten = [
        (f"01_ndvi_otsu", f"NDVI boven Otsu op maaiveld ({otsu:+.3f})", waarde > otsu),
        ("02_ndvi_min005", "NDVI boven -0.05, zeer ruim", waarde > -0.05),
        ("03_ndvi_000", "NDVI boven 0.00", waarde > 0.0),
        ("04_ndvi_005", "NDVI boven 0.05", waarde > 0.05),
        ("05_ndvi_010", "NDVI boven 0.10", waarde > 0.10),
        ("06_ndvi_015", "NDVI boven 0.15, streng", waarde > 0.15),
    ]
    for naam, drempel in (("07", otsu), ("08", 0.0), ("09", 0.05), ("10", 0.10)):
        kroon, veld = splits_kroon_en_veld(waarde, groen_op_ir, ir_k.maaiveld, drempel)
        varianten.append((f"{naam}_groenveld_{drempel:+.2f}".replace(".", ""), f"begroeid maaiveld bij NDVI {drempel:+.3f}", veld))
        varianten.append((f"{naam}_boomkroon_{drempel:+.2f}".replace(".", ""), f"kroon bij NDVI {drempel:+.3f}", kroon))

    return _schrijf(
        varianten, ir_k, analysevlak, "infrarood", instellingen.groen.min_oppervlakte_m2,
        instellingen.groen.vereenvoudiging_m, uitvoer_map / "infrarood.gpkg", uitsnede=infrarood,
    )
