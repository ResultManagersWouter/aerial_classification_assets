"""Is een gevonden groenvlak een boomkroon of echt groen op het maaiveld?

Boomkronen langs de weg komen er als wolkjes groen uit, en die horen niet thuis in een
vergelijking met geregistreerde groenvlakken: de kroon hangt boven verharding die gewoon
klopt. Kunnen we ze uit elkaar houden?

De aanname is dat een kroon er anders uitziet dan een gazon of een berm. Een kroon is
ruw, want blad en takken geven schaduwspikkels op korte afstand; een gazon is glad. Een
kroon is klein en rond; een berm is lang en smal en een plantsoen is groot. En een kroon
hangt vaak boven geregistreerde verharding, terwijl een groenvlak op geregistreerd groen
ligt. Die kenmerken zitten hieronder.

Geijkt wordt er op de stamlocaties uit het bomenregister van Amsterdam. Dat register is
geen perfecte waarheid - een boom in een park staat ook in een groenvlak - dus de labels
zijn zo gekozen dat ze de vraag stellen die ertoe doet: hangt dit groen boven verharding
met een stam eronder, of is het begroeid maaiveld.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from scipy import ndimage

from luchtfoto_objecten.modeldetectie import Beeldkenmerken

logger = logging.getLogger(__name__)

KENMERKEN = ["oppervlakte_m2", "compactheid", "breedte_m", "textuur", "exg", "helderheid"]
BOOMAFSTAND_M = 2.0
ZAAD = 0


def analyseer(gebied, instellingen=None, model: str = "regels"):
    """Haalt beeld, registratie en bomen op en onderzoekt of kroon en groenvlak te scheiden zijn.

    Geeft de gekenmerkte vlakken terug, een vergelijking van de mediaan per kenmerk, en de
    score van een model dat alleen op beeldkenmerken kijkt.
    """
    from luchtfoto_objecten.instellingen import Instellingen
    from luchtfoto_objecten.modeldetectie import (
        _kenmerken_van, groenmodellen, haal_beelden, naar_laag, opschonen,
    )
    from luchtfoto_objecten.referentie.amsterdam import AmsterdamRegistratie

    instellingen = instellingen or Instellingen.laden()
    hoofd, groenbeeld, registratie, analysevlak = haal_beelden(gebied, instellingen)
    kenmerken = _kenmerken_van(groenbeeld, registratie, analysevlak, instellingen)

    # Alleen het gevraagde model draaien. Via bouw_detectielagen kwam hier eerder het
    # volledige modellenpakket voorbij, inclusief clustering en gradient boosting over het
    # hele raster, om er vervolgens één laag uit te pakken.
    maskers = groenmodellen(kenmerken, instellingen)
    if model not in maskers:
        raise SystemExit(f"Onbekend groenmodel '{model}'. Beschikbaar: {', '.join(sorted(maskers))}")
    vlakken = naar_laag(
        opschonen(maskers[model] & kenmerken.maaiveld, kenmerken.transform, instellingen.groen.min_oppervlakte_m2),
        kenmerken.transform, "groen", model,
        instellingen.groen.min_oppervlakte_m2, instellingen.groen.vereenvoudiging_m, analysevlak,
    )
    if vlakken.empty:
        raise SystemExit(f"Geen groenvlakken van model '{model}' om te onderzoeken")
    verharding = pd.concat(
        [registratie["wegdelen"], registratie["onbegroeideterreindelen"]], ignore_index=True
    )
    bomen = AmsterdamRegistratie(cache_map=instellingen.cache_map).haal("bomen", gebied.bbox)
    logger.info("Bomenregister: %s stammen in dit gebied", len(bomen))

    tabel = kenmerk_vlakken(
        vlakken, kenmerken, gpd.GeoDataFrame(verharding, crs=vlakken.crs),
        registratie["begroeideterreindelen"], bomen,
    )
    tabel["label"] = label_vlakken(tabel)
    medianen, scores = onderzoek_scheidbaarheid(tabel)
    return tabel, medianen, scores


def kenmerk_vlakken(
    vlakken: gpd.GeoDataFrame, kenmerken: Beeldkenmerken, verharding: gpd.GeoDataFrame,
    begroeid: gpd.GeoDataFrame, bomen: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Vorm, ruwheid en ligging per gevonden groenvlak, plus het aantal stammen erin."""
    if vlakken.empty:
        return vlakken

    tabel = vlakken.copy().reset_index(drop=True)
    oppervlak = tabel.geometry.area
    omtrek = tabel.geometry.length.replace(0, np.nan)
    tabel["oppervlakte_m2"] = oppervlak.round(2)
    # Een cirkel haalt compactheid 1, een lange smalle berm blijft daar ver onder.
    tabel["compactheid"] = (4 * np.pi * oppervlak / (omtrek**2)).fillna(0).round(3)
    # Hydraulische breedte: voor een lange strook is dit ongeveer de breedte van de strook.
    tabel["breedte_m"] = (4 * oppervlak / omtrek).fillna(0).round(2)

    for naam, vlak in (("textuur", "textuur_1m"), ("exg", "exg"), ("helderheid", "helderheid")):
        tabel[naam] = _gemiddelde_per_vlak(tabel.geometry, kenmerken, vlak).round(4)

    tabel["aandeel_verharding"] = _overlapaandeel(tabel.geometry, verharding).round(3)
    tabel["aandeel_begroeid"] = _overlapaandeel(tabel.geometry, begroeid).round(3)
    tabel["stammen"] = _stammen_per_vlak(tabel.geometry, bomen)
    return tabel


def _gemiddelde_per_vlak(geometrieen: gpd.GeoSeries, kenmerken: Beeldkenmerken, vlaknaam: str) -> pd.Series:
    labels = rasterize(
        ((geo, index + 1) for index, geo in enumerate(geometrieen)),
        out_shape=kenmerken.vorm, transform=kenmerken.transform, fill=0, dtype=np.int32,
    )
    index = np.arange(1, len(geometrieen) + 1)
    waarden = ndimage.mean(kenmerken.vlakken[vlaknaam], labels=labels, index=index)
    return pd.Series(np.nan_to_num(waarden), index=geometrieen.index)


def _overlapaandeel(geometrieen: gpd.GeoSeries, ander: gpd.GeoDataFrame) -> pd.Series:
    if ander.empty:
        return pd.Series(0.0, index=geometrieen.index)
    index = ander.sindex
    aandelen = []
    for geo in geometrieen:
        posities = index.query(geo, predicate="intersects")
        if len(posities) == 0 or geo.area <= 0:
            aandelen.append(0.0)
            continue
        overlap = sum(geo.intersection(vorm).area for vorm in ander.geometry.iloc[posities])
        aandelen.append(min(overlap / geo.area, 1.0))
    return pd.Series(aandelen, index=geometrieen.index)


def _stammen_per_vlak(geometrieen: gpd.GeoSeries, bomen: gpd.GeoDataFrame) -> pd.Series:
    if bomen.empty:
        return pd.Series(0, index=geometrieen.index)
    index = bomen.sindex
    aantallen = []
    for geo in geometrieen:
        ruim = geo.buffer(BOOMAFSTAND_M)
        posities = index.query(ruim, predicate="intersects")
        aantallen.append(int(sum(ruim.contains(punt) for punt in bomen.geometry.iloc[posities])))
    return pd.Series(aantallen, index=geometrieen.index)


def label_vlakken(tabel: gpd.GeoDataFrame, drempel: float = 0.5) -> pd.Series:
    """De ijking: kroon boven verharding met een stam eronder, of begroeid terreindeel.

    Een boom midden in een park valt onder geen van beide en blijft zonder label. Dat is
    opzet: daar is de vraag niet interessant, want kroon en ondergrond zijn er allebei
    groen, en meenemen zou de meting alleen vertroebelen.
    """
    kroon = (tabel["stammen"] > 0) & (tabel["aandeel_verharding"] >= drempel)
    vlak = (tabel["aandeel_begroeid"] >= drempel) & (tabel["aandeel_verharding"] < 0.2)
    label = pd.Series(pd.NA, index=tabel.index, dtype="object")
    label[vlak] = "groenvlak"
    label[kroon] = "boomkroon"
    return label


def onderzoek_scheidbaarheid(tabel: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hoe goed zijn kroon en groenvlak uit elkaar te houden op beeldkenmerken alleen?

    Getoetst met kruisvalidatie, zodat het model zich niet op de eigen voorbeelden
    beoordeelt. Alleen beeldkenmerken, dus vorm, ruwheid en kleur. De ligging ten opzichte
    van de registratie zit er bewust niet in, want die zit ook in het label; meenemen zou
    betekenen dat we onszelf meten.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    gelabeld = tabel[tabel["label"].notna()]
    medianen = (
        gelabeld.groupby("label")[[*KENMERKEN, "stammen"]].median().round(3).reset_index()
        if not gelabeld.empty
        else pd.DataFrame()
    )
    telling = gelabeld["label"].value_counts()
    if len(telling) < 2 or telling.min() < 10:
        logger.warning("Te weinig gelabelde vlakken voor een eerlijke toets: %s", telling.to_dict())
        return medianen, pd.DataFrame()

    x = gelabeld[KENMERKEN].to_numpy(dtype=float)
    y = (gelabeld["label"] == "boomkroon").to_numpy()
    voorspeld = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=ZAAD), x, y, cv=5
    ).astype(bool)

    tp = int((y & voorspeld).sum())
    fp = int((~y & voorspeld).sum())
    fn = int((y & ~voorspeld).sum())
    tn = int((~y & ~voorspeld).sum())
    precisie = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # De bodem: altijd de grootste klasse roepen. Daar moet een model overheen komen.
    bodem = max(float(y.mean()), 1 - float(y.mean()))
    scores = pd.DataFrame([{
        "gelabelde_vlakken": int(len(gelabeld)),
        "boomkronen": int(y.sum()),
        "groenvlakken": int((~y).sum()),
        "juist": round((tp + tn) / len(y), 3),
        "juist_bij_altijd_grootste_klasse": round(bodem, 3),
        "precisie_boomkroon": round(precisie, 3),
        "recall_boomkroon": round(recall, 3),
        "f1_boomkroon": round(2 * precisie * recall / (precisie + recall), 3) if precisie + recall else 0.0,
    }])
    return medianen, scores
