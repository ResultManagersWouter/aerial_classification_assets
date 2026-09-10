"""Detectiemodellen naast elkaar zetten op één gebied.

Twee losse binaire vragen, elk op de opname die erbij hoort: groen op de nieuwste
opname met blad, verharding op de nieuwste 8cm-ortho. Per vraag draaien alle modellen
op precies dezelfde pixels, zodat de scores onderling vergelijkbaar zijn.

    altijd_positief  zegt overal ja, staat erin als bodem
    regels_ruw       de huidige drempels op kleur en textuur, zonder opschoning
    regels           diezelfde drempels met de morfologische opschoning erachter
    logistisch       logistische regressie op de pixelkenmerken
    randomforest     random forest op dezelfde kenmerken
    gradient_boost   histogram gradient boosting op dezelfde kenmerken

De BGT is de referentie. Let op wat dat betekent: de registratie is juist het bestand dat
we willen controleren, dus dit meet overeenstemming, geen waarheid. Een model dat hoger
scoort leest het beeld beter op de plekken waar de registratie klopt, en dan is de rest
van het verschil een echt signaal in plaats van modelruis.

Het westelijke deel van het gebied traint, het oostelijke deel wordt gescoord, zodat een
model niet gewoon de pixels kan onthouden die het al gezien heeft.

De klassen zijn scheef verdeeld: in de binnenstad is het maaiveld binnen de beheerkaart
voor ruim negentig procent verharding en nauwelijks begroeid. Daarom wordt er gebalanceerd
getraind, staat altijd_positief als bodem in de tabel, en sorteren we op MCC. Op IoU wint
anders het model dat overal ja zegt: dat haalt hier een IoU van 0,93 zonder iets te doen.

Vereist scikit-learn, zie requirements-ml.txt.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rasterio.transform import Affine
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from luchtfoto_objecten.bladstand import bepaal_groenbron
from luchtfoto_objecten.detectie.groen import _groendrempel, bepaal_groenmasker
from luchtfoto_objecten.detectie.kenmerken import (
    exces_groen,
    grijswaarde,
    hsv_kanalen,
    lokale_standaarddeviatie,
    naar_float,
    vari,
)
from luchtfoto_objecten.detectie.verharding import bepaal_verhardingsmasker
from luchtfoto_objecten.gebieden import Gebied
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.pipeline import _analysevlak
from luchtfoto_objecten.raster import meters_naar_pixels, polygonen_naar_masker
from luchtfoto_objecten.referentie.amsterdam import AmsterdamRegistratie
from luchtfoto_objecten.themas import bronnen_voor_analyse
from luchtfoto_objecten.wmts import LuchtfotoWMTS

logger = logging.getLogger(__name__)

TRAINPIXELS = 120_000
TESTPIXELS = 200_000
ZAAD = 0

KENMERKNAMEN = [
    "rood", "groen", "blauw", "exg", "vari", "tint", "verzadiging", "helderheid",
    "grijs", "textuur_1m", "textuur_3m",
]


def vergelijk_modellen(
    gebied: Gebied, instellingen: Instellingen | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scoort alle modellen op dit gebied en geeft de vergelijking en het kenmerkbelang terug."""
    instellingen = instellingen or Instellingen.laden()

    hoofd = _uitsnede(instellingen.luchtfoto.laag, instellingen.luchtfoto.zoom, gebied, instellingen)
    keuze = bepaal_groenbron(
        gebied.bbox,
        hoofdlaag=instellingen.luchtfoto.laag,
        zoom=instellingen.luchtfoto.zoom,
        cache_map=instellingen.cache_map,
        voorkeur=instellingen.luchtfoto.groenlaag,
        minimaal_aandeel=instellingen.luchtfoto.groen_minimaal_aandeel,
    )
    groenbeeld = (
        hoofd
        if (keuze.laag, keuze.zoom) == (hoofd.laag, hoofd.zoom)
        else _uitsnede(keuze.laag, keuze.zoom, gebied, instellingen)
    )

    registratie = AmsterdamRegistratie(cache_map=instellingen.cache_map).haal_meerdere(
        bronnen_voor_analyse(), gebied.bbox
    )
    analysevlak = _analysevlak(gebied, registratie, instellingen)

    groentabel, groenbelang = _vergelijk_klasse(
        "groen", groenbeeld, ("begroeideterreindelen",), ("wegdelen", "onbegroeideterreindelen"),
        registratie, analysevlak, instellingen,
    )
    verhardingtabel, verhardingbelang = _vergelijk_klasse(
        "verharding", hoofd, ("wegdelen", "onbegroeideterreindelen"), ("begroeideterreindelen",),
        registratie, analysevlak, instellingen,
    )

    vergelijking = pd.concat([groentabel, verhardingtabel], ignore_index=True)
    vergelijking.insert(1, "opname", np.where(vergelijking["klasse"] == "groen", groenbeeld.laag, hoofd.laag))
    return vergelijking, pd.concat([groenbelang, verhardingbelang], ignore_index=True)


def _uitsnede(laag: str, zoom: int, gebied: Gebied, instellingen: Instellingen):
    wmts = LuchtfotoWMTS(
        laag=laag, zoom=zoom, cache_map=instellingen.cache_map, max_werkers=instellingen.luchtfoto.max_werkers
    )
    return wmts.haal_uitsnede(gebied.bbox)


def _kenmerkvlakken(rgb: np.ndarray, transform: Affine):
    """Levert de kenmerken één vlak per keer op, in de volgorde van KENMERKNAMEN.

    Eén vlak per keer, want op zoom 15 is een gebied van een halve kilometer al tientallen
    miljoenen pixels en alle kenmerken tegelijk vasthouden kost meer dan een gigabyte.
    """
    beeld = naar_float(rgb)
    yield beeld[:, :, 0]
    yield beeld[:, :, 1]
    yield beeld[:, :, 2]
    del beeld
    yield exces_groen(rgb)
    yield vari(rgb)
    tint, verzadiging, helderheid = hsv_kanalen(rgb)
    yield tint
    yield verzadiging
    yield helderheid
    del tint, verzadiging, helderheid
    grijs = grijswaarde(rgb)
    yield grijs
    yield lokale_standaarddeviatie(grijs, meters_naar_pixels(1.0, transform))
    yield lokale_standaarddeviatie(grijs, meters_naar_pixels(3.0, transform))


def _kenmerkmatrix(rgb: np.ndarray, transform: Affine, *indexen: np.ndarray) -> list[np.ndarray]:
    """Bouwt per pixelselectie een kenmerkmatrix, zonder alle vlakken tegelijk te bewaren."""
    kolommen: list[list[np.ndarray]] = [[] for _ in indexen]
    for vlak in _kenmerkvlakken(rgb, transform):
        plat = np.ascontiguousarray(vlak, dtype=np.float32).ravel()
        for positie, index in enumerate(indexen):
            kolommen[positie].append(plat[index])
        del plat, vlak
    return [np.column_stack(kolom) for kolom in kolommen]


def _scores(waar: np.ndarray, voorspeld: np.ndarray) -> dict[str, float]:
    """IoU en precisie alleen zeggen te weinig bij scheve klassen, vandaar ook MCC."""
    tp = float(np.count_nonzero(waar & voorspeld))
    fp = float(np.count_nonzero(~waar & voorspeld))
    fn = float(np.count_nonzero(waar & ~voorspeld))
    tn = float(np.count_nonzero(~waar & ~voorspeld))
    precisie = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificiteit = tn / (tn + fp) if tn + fp else 0.0
    noemer = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "iou": round(tp / (tp + fp + fn), 3) if tp + fp + fn else 0.0,
        "precisie": round(precisie, 3),
        "recall": round(recall, 3),
        "specificiteit": round(specificiteit, 3),
        "gebalanceerd": round((recall + specificiteit) / 2, 3),
        "mcc": round((tp * tn - fp * fn) / noemer, 3) if noemer else 0.0,
        "voorspeld_aandeel": round(float(np.count_nonzero(voorspeld)) / voorspeld.size, 3),
    }


def _kies_pixels(geldig: np.ndarray, aantal: int, rng: np.random.Generator) -> np.ndarray:
    kandidaten = np.flatnonzero(geldig.ravel())
    if kandidaten.size <= aantal:
        return kandidaten
    return rng.choice(kandidaten, size=aantal, replace=False)


def _kies_gebalanceerd(geldig: np.ndarray, label: np.ndarray, aantal: int, rng: np.random.Generator) -> np.ndarray:
    """Evenveel pixels met als zonder de klasse, anders leert een model bij 93% verharding
    simpelweg altijd ja te zeggen."""
    positief = _kies_pixels(geldig & label, aantal // 2, rng)
    negatief = _kies_pixels(geldig & ~label, aantal // 2, rng)
    return np.concatenate([positief, negatief])


def _regelmaskers(klasse: str, rgb: np.ndarray, transform: Affine, instellingen: Instellingen, uitsluit: np.ndarray):
    """De huidige drempels, ruw en opgeschoond, zodat het effect van de morfologie zichtbaar wordt."""
    if klasse == "groen":
        exg = exces_groen(rgb)
        ruw = (exg > _groendrempel(exg, instellingen.groen.exg_ondergrens)) & ~uitsluit
        return ruw, bepaal_groenmasker(rgb, transform, instellingen.groen, uitsluit)

    params = instellingen.verharding
    _, verzadiging, helderheid = hsv_kanalen(rgb)
    textuur = lokale_standaarddeviatie(grijswaarde(rgb), meters_naar_pixels(1.0, transform))
    ruw = (
        (verzadiging <= params.max_verzadiging)
        & (helderheid >= params.min_helderheid)
        & (helderheid <= params.max_helderheid)
        & (textuur <= params.max_textuur)
        & ~uitsluit
    )
    return ruw, bepaal_verhardingsmasker(rgb, transform, params, uitsluit)


def _lerende_modellen() -> dict:
    return {
        "logistisch": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "randomforest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=ZAAD),
        "gradient_boost": HistGradientBoostingClassifier(max_iter=200, random_state=ZAAD),
    }


def _vergelijk_klasse(
    klasse: str, uitsnede, labelbronnen, tegenbronnen, registratie, analysevlak, instellingen
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vorm = uitsnede.afbeelding.shape[:2]
    transform = uitsnede.transform

    binnen = polygonen_naar_masker([analysevlak], vorm, transform)
    uitsluit = polygonen_naar_masker(
        registratie["panden"].geometry, vorm, transform, buffer_m=instellingen.verharding.gebouwbuffer_m
    ) | polygonen_naar_masker(registratie["waterdelen"].geometry, vorm, transform)

    label = np.zeros(vorm, dtype=bool)
    for bron in labelbronnen:
        label |= polygonen_naar_masker(registratie[bron].geometry, vorm, transform)
    tegen = np.zeros(vorm, dtype=bool)
    for bron in tegenbronnen:
        tegen |= polygonen_naar_masker(registratie[bron].geometry, vorm, transform)

    # Pixels die in beide klassen geregistreerd staan zeggen niets, die blijven buiten de score.
    geldig = binnen & ~uitsluit & ~(label & tegen)
    grens = vorm[1] // 2
    west, oost = geldig.copy(), geldig.copy()
    west[:, grens:] = False
    oost[:, :grens] = False
    logger.info(
        "%s: %.0f%% van het beeld bruikbaar, daarvan %.1f%% geregistreerd als %s",
        klasse, 100 * geldig.mean(), 100 * label[geldig].mean(), klasse,
    )

    ruw, opgeschoond = _regelmaskers(klasse, uitsnede.afbeelding, transform, instellingen, uitsluit)

    rng = np.random.default_rng(ZAAD)
    trainindex = _kies_gebalanceerd(west, label, TRAINPIXELS, rng)
    testindex = _kies_pixels(oost, TESTPIXELS, rng)
    kenmerken_train, kenmerken_test = _kenmerkmatrix(uitsnede.afbeelding, transform, trainindex, testindex)
    label_train, label_test = label.ravel()[trainindex], label.ravel()[testindex]

    resultaten = [
        {"model": "altijd_positief", **_scores(label_test, np.ones_like(label_test))},
        {"model": "regels_ruw", **_scores(label_test, ruw.ravel()[testindex])},
        {"model": "regels", **_scores(label_test, opgeschoond.ravel()[testindex])},
    ]
    belang = {}
    for naam, model in _lerende_modellen().items():
        model.fit(kenmerken_train, label_train)
        resultaten.append({"model": naam, **_scores(label_test, model.predict(kenmerken_test).astype(bool))})
        if hasattr(model, "feature_importances_"):
            belang[naam] = model.feature_importances_

    tabel = pd.DataFrame(resultaten)
    tabel.insert(0, "klasse", klasse)
    tabel.insert(2, "referentie_aandeel", round(float(label_test.mean()), 3))
    tabel = tabel.sort_values("mcc", ascending=False).reset_index(drop=True)

    belangtabel = pd.DataFrame(belang, index=KENMERKNAMEN).round(3)
    belangtabel.insert(0, "klasse", klasse)
    return tabel, belangtabel.reset_index(names="kenmerk")
