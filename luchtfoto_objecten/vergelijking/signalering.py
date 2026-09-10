from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from luchtfoto_objecten.geo_hulp import RD, lege_gdf
from luchtfoto_objecten.instellingen import SignaleringParameters

logger = logging.getLogger(__name__)

BEVESTIGD = "bevestigd"
AFWIJKENDE_GEOMETRIE = "afwijkende_geometrie"
NIET_ZICHTBAAR = "niet_zichtbaar_op_luchtfoto"
ONTBREEKT = "ontbreekt_in_registratie"
ANDERE_KLASSE = "geregistreerd_als_andere_klasse"
GEEN_REGISTRATIE = "geen_registratie_op_maaiveld"

SIGNAAL_KOLOMMEN = [
    "thema", "detectieklasse", "objectbron", "registratiebron", "status", "dekkingsgraad",
    "oppervlakte_m2", "afwijking_m2", "prioriteit", "identificatie", "toelichting",
]
WITTE_VLEK_KOLOMMEN = [
    "thema", "objectbron", "status", "oppervlakte_m2", "aandeel_groen",
    "aandeel_verharding", "vermoedelijke_klasse", "prioriteit", "toelichting",
]


def vergelijk_vlakken(
    detectie: gpd.GeoDataFrame,
    registratie: gpd.GeoDataFrame,
    params: SignaleringParameters,
    thema: str,
    detectieklasse: str,
    identificatieveld: str | None = None,
    detectie_voor_nieuw: gpd.GeoDataFrame | None = None,
    registratieset: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Zet de vlakken van de luchtfoto naast de geregistreerde vlakken van één thema.

    Met detectie_voor_nieuw geef je een afwijkende set mee voor de vraag welke vlakken
    ontbreken in de registratie. Zo voorkom je bijvoorbeeld dat winters gras, dat op de
    voorjaarsopname op bestrating lijkt, als onbekende verharding wordt gemeld.
    """
    nieuw = detectie if detectie_voor_nieuw is None else detectie_voor_nieuw
    records = _beoordeel_registratie(detectie, registratie, params, thema, detectieklasse, identificatieveld)
    records += _beoordeel_detectie(nieuw, registratie, params, thema, detectieklasse, registratieset)
    if not records:
        return lege_gdf(SIGNAAL_KOLOMMEN)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=RD)


def analyseer_vlakdekkendheid(
    analysevlak: BaseGeometry,
    registratie_union: BaseGeometry | None,
    detecties: dict[str, gpd.GeoDataFrame],
    params: SignaleringParameters,
    thema: str,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Zoekt maaiveld dat in geen enkel geregistreerd vlak van dit thema valt.

    Het analysevlak is het gebied zonder panden en zonder water, dus de openbare ruimte
    die een beheerregistratie in principe volledig zou moeten beschrijven.
    """
    totaal = float(analysevlak.area)
    if totaal <= 0:
        return lege_gdf(WITTE_VLEK_KOLOMMEN), {"analysevlak_m2": 0.0, "dekking_pct": 0.0, "witte_vlekken_m2": 0.0}

    gedekt = 0.0
    if registratie_union is not None and not registratie_union.is_empty:
        gedekt = float(analysevlak.intersection(registratie_union).area)
        rest = analysevlak.difference(registratie_union)
    else:
        rest = analysevlak
    rest = _zonder_snippers(rest, params.sliverbreedte_m)

    samenvatting = {
        "analysevlak_m2": round(totaal, 1),
        "gedekt_m2": round(gedekt, 1),
        "dekking_pct": round(100 * gedekt / totaal, 1),
        "witte_vlekken_m2": round(max(totaal - gedekt, 0.0), 1),
        "witte_vlekken_gemeld_m2": round(float(rest.area), 1),
    }

    records = []
    for deel in _als_lijst(rest):
        if deel.is_empty or deel.area < params.min_witte_vlek_m2:
            continue
        aandelen = {
            klasse: _overlapaandeel(deel, laag) for klasse, laag in detecties.items()
        }
        vermoedelijk = max(aandelen, key=aandelen.get) if aandelen else "onbekend"
        if aandelen.get(vermoedelijk, 0.0) < 0.25:
            vermoedelijk = "onbepaald"
        records.append({
            "thema": thema,
            "objectbron": "luchtfoto",
            "status": GEEN_REGISTRATIE,
            "oppervlakte_m2": round(float(deel.area), 2),
            "aandeel_groen": round(aandelen.get("groen", 0.0), 3),
            "aandeel_verharding": round(aandelen.get("verharding", 0.0), 3),
            "vermoedelijke_klasse": vermoedelijk,
            "prioriteit": _prioriteit(GEEN_REGISTRATIE, deel.area),
            "toelichting": "Maaiveld zonder geregistreerd vlak in dit thema",
            "geometry": deel,
        })

    if not records:
        return lege_gdf(WITTE_VLEK_KOLOMMEN), samenvatting
    vlekken = gpd.GeoDataFrame(records, geometry="geometry", crs=RD)
    samenvatting["aantal_witte_vlekken"] = int(len(vlekken))
    logger.info(
        "Vlakdekkendheid %s: %.1f%% gedekt, %s witte vlekken",
        thema, samenvatting["dekking_pct"], len(vlekken),
    )
    return vlekken, samenvatting


def raster_samenvatting(
    signaleringen: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    params: SignaleringParameters,
) -> gpd.GeoDataFrame:
    """Telt de signalen per rastercel, zodat je ziet waar de registratie het meest achterloopt."""
    cellen = _maak_raster(bbox, params.celgrootte_m)
    if signaleringen.empty:
        cellen["aantal_objecten"] = 0
        cellen["aantal_signalen"] = 0
        cellen["afwijkend_m2"] = 0.0
        cellen["aandeel_signalen"] = 0.0
        cellen["achterstand_verwacht"] = False
        return cellen

    punten = signaleringen.copy()
    punten["geometry"] = punten.geometry.representative_point()
    punten["is_signaal"] = punten["status"] != BEVESTIGD
    punten["signaal_m2"] = np.where(punten["is_signaal"], punten["oppervlakte_m2"].fillna(0.0), 0.0)

    gekoppeld = gpd.sjoin(punten, cellen, how="inner", predicate="within")
    telling = gekoppeld.groupby("index_right").agg(
        aantal_objecten=("is_signaal", "size"),
        aantal_signalen=("is_signaal", "sum"),
        afwijkend_m2=("signaal_m2", "sum"),
    )
    cellen = cellen.join(telling)
    cellen["aantal_objecten"] = cellen["aantal_objecten"].fillna(0).astype(int)
    cellen["aantal_signalen"] = cellen["aantal_signalen"].fillna(0).astype(int)
    cellen["afwijkend_m2"] = cellen["afwijkend_m2"].fillna(0.0).round(1)
    cellen["aandeel_signalen"] = np.where(
        cellen["aantal_objecten"] > 0, cellen["aantal_signalen"] / cellen["aantal_objecten"], 0.0
    ).round(3)
    cellen["achterstand_verwacht"] = cellen["aantal_signalen"] >= params.cel_signaaldrempel
    return cellen


def samenvatting_per_thema(signaleringen: gpd.GeoDataFrame) -> pd.DataFrame:
    if signaleringen.empty:
        return pd.DataFrame(columns=["thema", "status", "aantal", "oppervlakte_m2"])
    samenvatting = (
        signaleringen.groupby(["thema", "status"])
        .agg(aantal=("status", "size"), oppervlakte_m2=("oppervlakte_m2", "sum"))
        .reset_index()
    )
    samenvatting["oppervlakte_m2"] = samenvatting["oppervlakte_m2"].round(1)
    return samenvatting.sort_values(["thema", "aantal"], ascending=[True, False]).reset_index(drop=True)


def _beoordeel_registratie(detectie, registratie, params, thema, detectieklasse, identificatieveld) -> list[dict]:
    if registratie.empty:
        return []
    index = detectie.sindex if not detectie.empty else None
    records = []
    for positie, rij in registratie.iterrows():
        geometrie = rij.geometry
        if geometrie is None or geometrie.is_empty or geometrie.area < params.min_registratie_m2:
            continue
        overlap = _overlapoppervlak(geometrie, detectie, index)
        dekking = overlap / geometrie.area
        if dekking >= params.min_dekkingsgraad:
            status = BEVESTIGD
            toelichting = "Registratie komt overeen met het beeld"
        elif dekking >= params.lage_dekkingsgraad:
            status = AFWIJKENDE_GEOMETRIE
            toelichting = "Slechts een deel van het vlak is op de foto terug te zien"
        else:
            status = NIET_ZICHTBAAR
            toelichting = "Op deze plek toont de foto iets anders dan geregistreerd"
        afwijking = geometrie.area - overlap
        records.append({
            "thema": thema,
            "detectieklasse": detectieklasse,
            "objectbron": "registratie",
            "registratiebron": rij.get("registratiebron"),
            "status": status,
            "dekkingsgraad": round(float(dekking), 3),
            "oppervlakte_m2": round(float(geometrie.area), 2),
            "afwijking_m2": round(float(afwijking), 2),
            "prioriteit": _prioriteit(status, afwijking),
            "identificatie": _identificatie(rij, identificatieveld, positie),
            "toelichting": toelichting,
            "geometry": geometrie,
        })
    return records


def _beoordeel_detectie(detectie, registratie, params, thema, detectieklasse, registratieset) -> list[dict]:
    if detectie.empty:
        return []
    index = registratie.sindex if not registratie.empty else None
    set_index = registratieset.sindex if registratieset is not None and not registratieset.empty else None
    records = []
    for positie, rij in detectie.iterrows():
        geometrie = rij.geometry
        if geometrie is None or geometrie.is_empty or geometrie.area < params.min_nieuw_oppervlakte_m2:
            continue
        overlap = _overlapoppervlak(geometrie, registratie, index)
        dekking = overlap / geometrie.area
        if dekking >= params.lage_dekkingsgraad:
            continue

        status = ONTBREEKT
        toelichting = "Op de foto zichtbaar, niet aanwezig in de registratie"
        if set_index is not None:
            setdekking = _overlapoppervlak(geometrie, registratieset, set_index) / geometrie.area
            if setdekking >= params.min_dekkingsgraad:
                status = ANDERE_KLASSE
                toelichting = "Wel geregistreerd, maar in een ander thema dan de foto laat zien"
        records.append({
            "thema": thema,
            "detectieklasse": detectieklasse,
            "objectbron": "luchtfoto",
            "registratiebron": None,
            "status": status,
            "dekkingsgraad": round(float(dekking), 3),
            "oppervlakte_m2": round(float(geometrie.area), 2),
            "afwijking_m2": round(float(geometrie.area - overlap), 2),
            "prioriteit": _prioriteit(status, geometrie.area - overlap),
            "identificatie": f"detectie_{positie}",
            "toelichting": toelichting,
            "geometry": geometrie,
        })
    return records


def _overlapoppervlak(geometrie, laag: gpd.GeoDataFrame, index) -> float:
    if laag.empty or index is None:
        return 0.0
    posities = index.query(geometrie, predicate="intersects")
    if len(posities) == 0:
        return 0.0
    delen = [
        geometrie.intersection(andere)
        for andere in laag.geometry.iloc[posities]
        if andere is not None and not andere.is_empty
    ]
    delen = [deel for deel in delen if not deel.is_empty]
    if not delen:
        return 0.0
    return min(float(unary_union(delen).area), float(geometrie.area))


def _overlapaandeel(geometrie, laag: gpd.GeoDataFrame) -> float:
    if laag.empty or geometrie.area <= 0:
        return 0.0
    return _overlapoppervlak(geometrie, laag, laag.sindex) / geometrie.area


def _zonder_snippers(geometrie: BaseGeometry, breedte_m: float) -> BaseGeometry:
    """Haalt randstroken weg die alleen ontstaan doordat twee bestanden anders zijn ingemeten.

    Twee vlakdekkende bestanden over hetzelfde gebied leveren altijd lange, flinterdunne
    verschillen op langs gedeelde randen. Een negatieve buffer gevolgd door een positieve
    laat alleen de verschillen over die echt ergens over gaan.
    """
    if geometrie.is_empty or breedte_m <= 0:
        return geometrie
    gekrompen = geometrie.buffer(-breedte_m / 2)
    if gekrompen.is_empty:
        return gekrompen
    return gekrompen.buffer(breedte_m / 2).intersection(geometrie)


def _als_lijst(geometrie: BaseGeometry) -> list[BaseGeometry]:
    if geometrie.is_empty:
        return []
    if hasattr(geometrie, "geoms"):
        return [deel for deel in geometrie.geoms if deel.geom_type in ("Polygon", "MultiPolygon")]
    return [geometrie]


def _identificatie(rij, veld: str | None, positie) -> str:
    if veld and veld in rij and rij[veld] is not None:
        return str(rij[veld])
    for kandidaat in ("identificatie", "imgeoIdentificatie", "guid", "id"):
        if kandidaat in rij and rij[kandidaat] is not None:
            return str(rij[kandidaat])
    return str(positie)


def _prioriteit(status: str, afwijking_m2: float) -> str:
    if status == BEVESTIGD:
        return "laag"
    if afwijking_m2 >= 100:
        return "hoog"
    if afwijking_m2 >= 25:
        return "midden"
    return "laag"


def _maak_raster(bbox: tuple[float, float, float, float], celgrootte: float) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = bbox
    cellen = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            cellen.append(box(x, y, min(x + celgrootte, xmax), min(y + celgrootte, ymax)))
            y += celgrootte
        x += celgrootte
    return gpd.GeoDataFrame({"celnummer": range(len(cellen))}, geometry=cellen, crs=RD)
