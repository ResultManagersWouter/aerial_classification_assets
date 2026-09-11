"""Per model een eigen laag met gedetecteerde vlakken, om ze visueel te vergelijken.

De modelvergelijking in modelvergelijking.py geeft cijfers, maar cijfers tegen de BGT
zeggen weinig zolang de BGT zelf de vraag is. Deze module levert de vlakken zelf, per
model een laag, zodat je in QGIS over de luchtfoto kunt kijken welk model het beeld
het beste volgt.

Elk model levert een masker over hetzelfde beeld, krijgt daarna dezelfde opschoning en
dezelfde minimale oppervlakte, en wordt op hetzelfde analysevlak geknipt. Wat overblijft
is dus echt het verschil tussen de modellen en niet tussen de nabewerking.

Over de groenmodellen, want daar zit het probleem. De huidige regels bepalen hun drempel
met Otsu over het hele beeld, dus inclusief daken en water, en pas daarna gaat het
uitsluitmasker eroverheen. Otsu zoekt dan de scheiding tussen donkere daken en de rest in
plaats van tussen groen en verharding, en dat legt de drempel te hoog: op een blok in
Centrum kwam hij op 0,098 uit terwijl de ondergrens in de configuratie 0,035 is. Vandaar
de varianten die hun drempel alleen op het maaiveld bepalen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, opening, remove_small_holes, remove_small_objects

from luchtfoto_objecten.detectie.kenmerken import (
    exces_groen,
    grijswaarde,
    hsv_kanalen,
    lokale_standaarddeviatie,
    naar_float,
    vari,
)
from luchtfoto_objecten.geo_hulp import RD, lege_gdf
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.raster import masker_naar_polygonen, meters_naar_pixels, pixeloppervlak, polygonen_naar_masker

logger = logging.getLogger(__name__)

KENMERKNAMEN = [
    "rood", "groen", "blauw", "exg", "vari", "gli", "tint", "verzadiging", "helderheid",
    "grijs", "textuur_1m", "textuur_3m",
]
BANDHOOGTE = 512
MONSTERPIXELS = 100_000
ZAAD = 0


@dataclass
class Beeldkenmerken:
    """De kenmerkvlakken van één opname, plus de maskers die erbij horen."""

    transform: Affine
    vlakken: dict[str, np.ndarray]
    uitsluit: np.ndarray
    maaiveld: np.ndarray

    @property
    def vorm(self) -> tuple[int, int]:
        return self.uitsluit.shape

    def matrix(self, rijen: slice) -> np.ndarray:
        return np.column_stack([self.vlakken[naam][rijen].ravel() for naam in KENMERKNAMEN])


def bereken_kenmerken(rgb: np.ndarray, transform: Affine, uitsluit: np.ndarray, maaiveld: np.ndarray):
    beeld = naar_float(rgb)
    rood, groen, blauw = beeld[:, :, 0], beeld[:, :, 1], beeld[:, :, 2]
    noemer = 2 * groen + rood + blauw
    grijs = grijswaarde(rgb)
    tint, verzadiging, helderheid = hsv_kanalen(rgb)
    vlakken = {
        "rood": rood, "groen": groen, "blauw": blauw,
        "exg": exces_groen(rgb),
        "vari": vari(rgb),
        "gli": np.where(noemer > 1e-6, (2 * groen - rood - blauw) / np.where(noemer > 1e-6, noemer, 1.0), 0.0),
        "tint": tint, "verzadiging": verzadiging, "helderheid": helderheid, "grijs": grijs,
        "textuur_1m": lokale_standaarddeviatie(grijs, meters_naar_pixels(1.0, transform)),
        "textuur_3m": lokale_standaarddeviatie(grijs, meters_naar_pixels(3.0, transform)),
    }
    return Beeldkenmerken(
        transform=transform,
        vlakken={naam: np.ascontiguousarray(vlak, dtype=np.float32) for naam, vlak in vlakken.items()},
        uitsluit=uitsluit,
        maaiveld=maaiveld,
    )


def _otsu_op_maaiveld(waarden: np.ndarray, maaiveld: np.ndarray, ondergrens: float) -> float:
    """Otsu alleen over het maaiveld, dus zonder daken en water die de drempel omhoog trekken."""
    monster = waarden[maaiveld]
    monster = monster[np.isfinite(monster)]
    if monster.size < 100:
        return ondergrens
    try:
        return float(threshold_otsu(monster))
    except ValueError:
        return ondergrens


def _otsu_op_heel_beeld(waarden: np.ndarray, ondergrens: float) -> float:
    try:
        return max(float(threshold_otsu(waarden[np.isfinite(waarden)])), ondergrens)
    except ValueError:
        return ondergrens


def groenmodellen(k: Beeldkenmerken, instellingen: Instellingen) -> dict[str, np.ndarray]:
    ondergrens = instellingen.groen.exg_ondergrens
    exg, varigetal, gli = k.vlakken["exg"], k.vlakken["vari"], k.vlakken["gli"]

    drempel_heel = _otsu_op_heel_beeld(exg, ondergrens)
    drempel_maaiveld = _otsu_op_maaiveld(exg, k.maaiveld, ondergrens)
    logger.info(
        "Groendrempel op excess green: %.4f over het hele beeld, %.4f op alleen het maaiveld",
        drempel_heel, drempel_maaiveld,
    )
    return {
        "regels": exg > drempel_heel,
        "exg_ondergrens": exg > ondergrens,
        "exg_maaiveld": exg > drempel_maaiveld,
        "vari_maaiveld": varigetal > _otsu_op_maaiveld(varigetal, k.maaiveld, 0.0),
        "gli_maaiveld": gli > _otsu_op_maaiveld(gli, k.maaiveld, 0.0),
        "hsv_groen": (k.vlakken["tint"] > 0.15) & (k.vlakken["tint"] < 0.45) & (k.vlakken["verzadiging"] > 0.12),
    }


def verhardingsmodellen(
    k: Beeldkenmerken, instellingen: Instellingen, groen: np.ndarray | None
) -> dict[str, np.ndarray]:
    params = instellingen.verharding
    regels = (
        (k.vlakken["verzadiging"] <= params.max_verzadiging)
        & (k.vlakken["helderheid"] >= params.min_helderheid)
        & (k.vlakken["helderheid"] <= params.max_helderheid)
        & (k.vlakken["textuur_1m"] <= params.max_textuur)
    )
    modellen = {"regels": regels}
    if groen is not None:
        # Verharding is in de openbare ruimte vooral "het maaiveld dat niet begroeid is",
        # dus het complement van het groen is een serieuze kandidaat en geen truc.
        modellen["niet_groen"] = k.maaiveld & ~groen
    return modellen


def _clusters(k: Beeldkenmerken, aantal: int, rng: np.random.Generator) -> np.ndarray:
    """Deelt het beeld ongesuperviseerd op in kleurgroepen, per band voorspeld."""
    from sklearn.cluster import KMeans

    posities = np.flatnonzero(k.maaiveld.ravel())
    if posities.size == 0:
        return np.zeros(k.vorm, dtype=np.int32)
    keuze = rng.choice(posities, size=min(MONSTERPIXELS, posities.size), replace=False)
    monster = np.column_stack([k.vlakken[naam].ravel()[keuze] for naam in KENMERKNAMEN])

    model = KMeans(n_clusters=aantal, n_init=4, random_state=ZAAD).fit(monster)
    labels = np.zeros(k.vorm, dtype=np.int32)
    for begin in range(0, k.vorm[0], BANDHOOGTE):
        rijen = slice(begin, min(begin + BANDHOOGTE, k.vorm[0]))
        labels[rijen] = model.predict(k.matrix(rijen)).reshape(labels[rijen].shape)
    return labels


def kmeans_groen(k: Beeldkenmerken, ondergrens: float, aantal: int = 6) -> np.ndarray:
    """De kleurgroepen waarvan het gemiddelde groensignaal boven de ondergrens ligt."""
    labels = _clusters(k, aantal, np.random.default_rng(ZAAD))
    masker = np.zeros(k.vorm, dtype=bool)
    for cluster in range(aantal):
        hoort_bij = labels == cluster
        if hoort_bij.any() and float(k.vlakken["exg"][hoort_bij].mean()) > ondergrens:
            masker |= hoort_bij
    return masker


def leer_op_registratie(k: Beeldkenmerken, label: np.ndarray, aantal: int = 120_000) -> np.ndarray:
    """Histogram gradient boosting, getraind op de BGT als label.

    Dit model leert dus de registratie na, inclusief de plekken waar die achterloopt. Het
    staat er juist in als tegenhanger: waar dit model afwijkt van de beeldmodellen zie je
    het verschil tussen wat er geregistreerd is en wat er te zien is.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(ZAAD)
    positief = np.flatnonzero((k.maaiveld & label).ravel())
    negatief = np.flatnonzero((k.maaiveld & ~label).ravel())
    if positief.size < 100 or negatief.size < 100:
        logger.warning("Te weinig voorbeelden om op de registratie te leren, laag blijft leeg")
        return np.zeros(k.vorm, dtype=bool)

    keuze = np.concatenate([
        rng.choice(positief, size=min(aantal // 2, positief.size), replace=False),
        rng.choice(negatief, size=min(aantal // 2, negatief.size), replace=False),
    ])
    kenmerken = np.column_stack([k.vlakken[naam].ravel()[keuze] for naam in KENMERKNAMEN])
    model = HistGradientBoostingClassifier(max_iter=200, random_state=ZAAD).fit(kenmerken, label.ravel()[keuze])

    masker = np.zeros(k.vorm, dtype=bool)
    for begin in range(0, k.vorm[0], BANDHOOGTE):
        rijen = slice(begin, min(begin + BANDHOOGTE, k.vorm[0]))
        masker[rijen] = model.predict(k.matrix(rijen)).reshape(masker[rijen].shape)
    return masker


def opschonen(masker: np.ndarray, transform: Affine, min_oppervlakte_m2: float) -> np.ndarray:
    """Dezelfde nabewerking voor elk model, zodat het verschil in de modellen zit."""
    min_pixels = max(4, int(min_oppervlakte_m2 / pixeloppervlak(transform)))
    masker = opening(masker, disk(1))
    masker = remove_small_objects(masker, max_size=min_pixels)
    masker = remove_small_holes(masker, max_size=min_pixels)
    return closing(masker, disk(1))


def naar_laag(
    masker: np.ndarray, transform: Affine, klasse: str, model: str,
    min_oppervlakte_m2: float, vereenvoudiging_m: float, analysevlak: BaseGeometry,
) -> gpd.GeoDataFrame:
    geometrieen = masker_naar_polygonen(masker, transform, min_oppervlakte_m2, vereenvoudiging_m)
    if not geometrieen:
        return lege_gdf(["klasse", "model", "oppervlakte_m2"])
    laag = gpd.GeoDataFrame(
        {"klasse": klasse, "model": model, "oppervlakte_m2": [round(geo.area, 2) for geo in geometrieen]},
        geometry=geometrieen, crs=RD,
    )
    laag = gpd.clip(laag, analysevlak)
    laag = laag[~laag.geometry.is_empty & laag.geometry.notna()]
    laag = laag[laag.geometry.area >= min_oppervlakte_m2]
    laag["oppervlakte_m2"] = laag.geometry.area.round(2)
    return laag.reset_index(drop=True)


def haal_beelden(gebied, instellingen: Instellingen):
    """De twee opnamen, de registratie en het analysevlak, zoals de pijplijn ze ook gebruikt.

    Alles komt uit dezelfde cache als een gewone analyse, dus dit kost geen extra downloads
    als het gebied al eens gedraaid is.
    """
    from luchtfoto_objecten.bladstand import bepaal_groenbron
    from luchtfoto_objecten.pipeline import _analysevlak
    from luchtfoto_objecten.referentie.amsterdam import AmsterdamRegistratie
    from luchtfoto_objecten.themas import bronnen_voor_analyse
    from luchtfoto_objecten.wmts import LuchtfotoWMTS

    def uitsnede_van(laag: str, zoom: int):
        wmts = LuchtfotoWMTS(
            laag=laag, zoom=zoom, cache_map=instellingen.cache_map, max_werkers=instellingen.luchtfoto.max_werkers
        )
        return wmts.haal_uitsnede(gebied.bbox)

    hoofd = uitsnede_van(instellingen.luchtfoto.laag, instellingen.luchtfoto.zoom)
    keuze = bepaal_groenbron(
        gebied.bbox,
        hoofdlaag=instellingen.luchtfoto.laag,
        zoom=instellingen.luchtfoto.zoom,
        cache_map=instellingen.cache_map,
        voorkeur=instellingen.luchtfoto.groenlaag,
        minimaal_aandeel=instellingen.luchtfoto.groen_minimaal_aandeel,
    )
    groenbeeld = (
        hoofd if (keuze.laag, keuze.zoom) == (hoofd.laag, hoofd.zoom) else uitsnede_van(keuze.laag, keuze.zoom)
    )
    registratie = AmsterdamRegistratie(cache_map=instellingen.cache_map).haal_meerdere(
        bronnen_voor_analyse(), gebied.bbox
    )
    return hoofd, groenbeeld, registratie, _analysevlak(gebied, registratie, instellingen)


def bouw_detectielagen(gebied, instellingen: Instellingen | None = None):
    """Haalt het beeld op en levert per model een laag met gedetecteerde vlakken."""
    from luchtfoto_objecten.infrarood import haal_infrarood

    instellingen = instellingen or Instellingen.laden()
    hoofd, groenbeeld, registratie, analysevlak = haal_beelden(gebied, instellingen)
    return detectielagen(
        groenbeeld, hoofd, registratie, analysevlak, instellingen,
        infrarood=haal_infrarood(gebied, instellingen),
    )


def detectielagen(
    groenbeeld, hoofd, registratie: dict[str, gpd.GeoDataFrame], analysevlak: BaseGeometry,
    instellingen: Instellingen, infrarood=None,
) -> tuple[dict[str, gpd.GeoDataFrame], pd.DataFrame]:
    """Per klasse en per model een laag, plus een tabel met wat elk model oplevert."""
    lagen: dict[str, gpd.GeoDataFrame] = {}
    overzicht: list[dict] = []

    groen_kenmerken = _kenmerken_van(groenbeeld, registratie, analysevlak, instellingen)
    maskers = groenmodellen(groen_kenmerken, instellingen)
    maskers["kmeans"] = kmeans_groen(groen_kenmerken, instellingen.groen.exg_ondergrens)
    maskers["boosting_bgt"] = leer_op_registratie(
        groen_kenmerken, _labelmasker(registratie, ("begroeideterreindelen",), groen_kenmerken)
    )

    groenmaskers: dict[str, np.ndarray] = {}
    for model, masker in maskers.items():
        schoon = opschonen(
            masker & groen_kenmerken.maaiveld, groen_kenmerken.transform, instellingen.groen.min_oppervlakte_m2
        )
        groenmaskers[model] = schoon
        lagen[f"detectie_groen_{model}"], regel = _laag_en_regel(
            schoon, groen_kenmerken, "groen", model, instellingen.groen.min_oppervlakte_m2,
            instellingen.groen.vereenvoudiging_m, analysevlak, groenbeeld.laag,
        )
        overzicht.append(regel)

    verharding_kenmerken = _kenmerken_van(hoofd, registratie, analysevlak, instellingen)
    groen_op_hoofd = _naar_ander_raster(groenmaskers.get("exg_maaiveld"), groen_kenmerken, verharding_kenmerken)
    maskers = verhardingsmodellen(verharding_kenmerken, instellingen, groen_op_hoofd)
    maskers["kmeans"] = _kmeans_verharding(verharding_kenmerken, instellingen)
    maskers["boosting_bgt"] = leer_op_registratie(
        verharding_kenmerken,
        _labelmasker(registratie, ("wegdelen", "onbegroeideterreindelen"), verharding_kenmerken),
    )
    verhardingsmaskers: dict[str, np.ndarray] = {}
    for model, masker in maskers.items():
        schoon = opschonen(
            masker & verharding_kenmerken.maaiveld, verharding_kenmerken.transform,
            instellingen.verharding.min_oppervlakte_m2,
        )
        verhardingsmaskers[model] = schoon
        lagen[f"detectie_verharding_{model}"], regel = _laag_en_regel(
            schoon, verharding_kenmerken, "verharding", model, instellingen.verharding.min_oppervlakte_m2,
            instellingen.verharding.vereenvoudiging_m, analysevlak, hoofd.laag,
        )
        overzicht.append(regel)

    # Kroon en maaiveld scheiden. De verharding komt van de voorjaarsopname, want die is
    # met kale bomen gevlogen en laat dus de grond onder de zomerse kroon zien.
    from luchtfoto_objecten.groenstructuur import bouw_lagen

    verhard_eronder = _naar_ander_raster(
        verhardingsmaskers.get("regels"), verharding_kenmerken, groen_kenmerken
    )
    if verhard_eronder is None:
        verhard_eronder = np.zeros(groen_kenmerken.vorm, dtype=bool)
    structuurlagen, structuuroverzicht = bouw_lagen(
        groen_kenmerken, groenmaskers["hsv_groen"], verhard_eronder, analysevlak, instellingen
    )
    lagen.update(structuurlagen)
    for regel in structuuroverzicht.to_dict("records"):
        overzicht.append({
            "klasse": regel["laag"].replace("detectie_", ""), "model": "groenstructuur",
            "opname": f"{groenbeeld.laag} + {hoofd.laag}", "vlakken": regel["vlakken"],
            "oppervlakte_m2": regel["oppervlakte_m2"], "aandeel_analysevlak": regel["aandeel_analysevlak"],
        })

    if infrarood is not None:
        lagen.update(
            _infraroodlagen(
                infrarood, groen_kenmerken, groenmaskers["hsv_groen"], registratie, analysevlak,
                instellingen, overzicht,
            )
        )

    return lagen, pd.DataFrame(overzicht)


def _infraroodlagen(
    infrarood, groen_kenmerken, groen_zomer, registratie, analysevlak, instellingen, overzicht,
) -> dict[str, gpd.GeoDataFrame]:
    """NDVI op de voorjaarsopname, en daarmee kroon los van begroeid maaiveld.

    Het infrarood is in het vroege voorjaar gevlogen met kale bomen, dus NDVI kijkt hier
    door de kroon heen naar de grond. Wat in de zomer groen is maar in het voorjaar geen
    begroeiing toont, is dus kroon boven iets anders.
    """
    from luchtfoto_objecten.infrarood import ndvi, ndvi_drempel, splits_kroon_en_veld

    ir_kenmerken = _kenmerken_van(infrarood, registratie, analysevlak, instellingen)
    waarde = ndvi(infrarood)
    drempel = ndvi_drempel(waarde, ir_kenmerken.maaiveld)
    logger.info("NDVI-drempel op het maaiveld: %+.3f (%s)", drempel, infrarood.laag)

    groen_op_ir = _naar_ander_raster(groen_zomer, groen_kenmerken, ir_kenmerken)
    if groen_op_ir is None:
        groen_op_ir = np.zeros(ir_kenmerken.vorm, dtype=bool)
    kroon, veld = splits_kroon_en_veld(waarde, groen_op_ir, ir_kenmerken.maaiveld, drempel)

    # Alleen groenveld en kroon. Een aparte NDVI-groenlaag zou per constructie hetzelfde
    # masker zijn als groenveld, en twee namen voor één laag maakt het beoordelen alleen
    # maar verwarrend.
    lagen = {}
    for laagnaam, masker, klasse, min_opp in (
        ("detectie_groenveld_ir", veld, "groenveld", instellingen.groenstructuur.strook_min_oppervlakte_m2),
        ("detectie_boomkroon_ir", kroon, "boomkroon", instellingen.groenstructuur.kroon_min_oppervlakte_m2),
    ):
        schoon = opschonen(masker, ir_kenmerken.transform, min_opp)
        laag, regel = _laag_en_regel(
            schoon, ir_kenmerken, klasse, "ndvi_infrarood", min_opp,
            instellingen.groenstructuur.vereenvoudiging_m, analysevlak, infrarood.laag,
        )
        lagen[laagnaam] = laag
        overzicht.append(regel)
    return lagen


def _laag_en_regel(masker, kenmerken, klasse, model, min_opp, vereenvoudiging, analysevlak, opname):
    laag = naar_laag(masker, kenmerken.transform, klasse, model, min_opp, vereenvoudiging, analysevlak)
    oppervlak = float(laag.geometry.area.sum()) if not laag.empty else 0.0
    regel = {
        "klasse": klasse,
        "model": model,
        "opname": opname,
        "vlakken": int(len(laag)),
        "oppervlakte_m2": round(oppervlak, 1),
        "aandeel_analysevlak": round(oppervlak / analysevlak.area, 3) if analysevlak.area else 0.0,
    }
    logger.info("%s/%s: %s vlakken, %.0f m2", klasse, model, regel["vlakken"], oppervlak)
    return laag, regel


def _kenmerken_van(uitsnede, registratie, analysevlak, instellingen) -> Beeldkenmerken:
    vorm = uitsnede.afbeelding.shape[:2]
    uitsluit = polygonen_naar_masker(
        registratie["panden"].geometry, vorm, uitsnede.transform, buffer_m=instellingen.verharding.gebouwbuffer_m
    ) | polygonen_naar_masker(registratie["waterdelen"].geometry, vorm, uitsnede.transform)
    maaiveld = polygonen_naar_masker([analysevlak], vorm, uitsnede.transform) & ~uitsluit
    return bereken_kenmerken(uitsnede.afbeelding, uitsnede.transform, uitsluit, maaiveld)


def _labelmasker(registratie, bronnen, kenmerken: Beeldkenmerken) -> np.ndarray:
    masker = np.zeros(kenmerken.vorm, dtype=bool)
    for bron in bronnen:
        masker |= polygonen_naar_masker(registratie[bron].geometry, kenmerken.vorm, kenmerken.transform)
    return masker


def _kmeans_verharding(k: Beeldkenmerken, instellingen: Instellingen, aantal: int = 6) -> np.ndarray:
    """De kleurgroepen die vlak en onverzadigd zijn, dus bestrating en asfalt."""
    labels = _clusters(k, aantal, np.random.default_rng(ZAAD))
    masker = np.zeros(k.vorm, dtype=bool)
    for cluster in range(aantal):
        hoort_bij = labels == cluster
        if not hoort_bij.any():
            continue
        groenachtig = float(k.vlakken["exg"][hoort_bij].mean()) > instellingen.groen.exg_ondergrens
        ruw = float(k.vlakken["textuur_1m"][hoort_bij].mean()) > instellingen.verharding.max_textuur
        if not groenachtig and not ruw:
            masker |= hoort_bij
    return masker


def _naar_ander_raster(masker, van: Beeldkenmerken, naar: Beeldkenmerken) -> np.ndarray | None:
    """Zet een masker over op het raster van een andere opname.

    De zomeropname en de 8cm-ortho hebben een andere resolutie, dus dat gaat via de
    polygonen in plaats van via de pixels.
    """
    if masker is None:
        return None
    geometrieen = masker_naar_polygonen(masker, van.transform, 1.0, 0.0)
    if not geometrieen:
        return np.zeros(naar.vorm, dtype=bool)
    # Zonder all_touched, anders groeit het masker bij elke oversteek met een pixel aan
    # elke rand, en dat stapelt op als je het vaker doet.
    return polygonen_naar_masker(geometrieen, naar.vorm, naar.transform, all_touched=False)
