from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from luchtfoto_objecten.bladstand import Bladstandkeuze, bepaal_groenbron, meet_groenaandeel
from luchtfoto_objecten.detectie.groen import bepaal_groenmasker, detecteer_groen
from luchtfoto_objecten.detectie.verharding import bepaal_verhardingsmasker, detecteer_verharding
from luchtfoto_objecten.gebieden import Gebied
from luchtfoto_objecten.geo_hulp import RD, lege_gdf
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.raster import polygonen_naar_masker, schrijf_geotiff
from luchtfoto_objecten.referentie.amsterdam import AmsterdamRegistratie
from luchtfoto_objecten.themas import GROEN, REGISTRATIESETS, VERHARDING, THEMAS, bronnen_voor_analyse, themas_van_set
from luchtfoto_objecten.uitvoer import schrijf_geojson, schrijf_geopackage, schrijf_samenvatting
from luchtfoto_objecten.vergelijking.signalering import (
    SIGNAAL_KOLOMMEN,
    WITTE_VLEK_KOLOMMEN,
    analyseer_vlakdekkendheid,
    raster_samenvatting,
    samenvatting_per_thema,
    vergelijk_vlakken,
)
from luchtfoto_objecten.wmts import LuchtfotoWMTS, Uitsnede

logger = logging.getLogger(__name__)


@dataclass
class AnalyseResultaat:
    gebied: Gebied
    lagen: dict[str, gpd.GeoDataFrame]
    samenvatting: dict
    per_thema: pd.DataFrame
    geotiff: Path | None = None
    geopackage: Path | None = None
    geojson_paden: list[Path] = field(default_factory=list)

    def toon(self) -> str:
        foto = self.samenvatting["luchtfoto"]
        regels = [
            f"Gebied      : {self.gebied}",
            f"Luchtfoto   : {foto['laag']} op zoom {foto['zoom']} ({foto['resolutie_m']} m/px)",
            f"Groenbeeld  : {foto['groenlaag']} ({foto['groenaandeel_pct']}% groene pixels)",
            "Detectie    : " + ", ".join(f"{sleutel}={waarde}" for sleutel, waarde in self.samenvatting["detectie"].items()),
            "Signalen    : " + ", ".join(f"{sleutel}={waarde}" for sleutel, waarde in self.samenvatting["signalen"].items()),
        ]
        for naam, dekking in self.samenvatting["vlakdekkendheid"].items():
            regels.append(
                f"Dekking {naam:<18}: {dekking['dekking_pct']:5.1f}% van het maaiveld, "
                f"{dekking.get('aantal_witte_vlekken', 0)} witte vlekken van samen "
                f"{dekking.get('witte_vlekken_gemeld_m2', 0):.0f} m2"
            )
        regels.append(
            f"Achterstand : {self.samenvatting['cellen_met_achterstand']} van "
            f"{self.samenvatting['totaal_cellen']} rastercellen boven de drempel"
        )
        return "\n".join(regels)


def voer_analyse_uit(
    gebied: Gebied,
    instellingen: Instellingen | None = None,
    uitvoer_map: Path | None = None,
    schrijf_bestanden: bool = True,
    sam_checkpoint: str | None = None,
    sam_modeltype: str = "vit_b",
    toon_voortgang: bool = True,
) -> AnalyseResultaat:
    instellingen = instellingen or Instellingen.laden()
    uitvoer_map = uitvoer_map or (instellingen.uitvoer_map / gebied.naam)
    logger.info("Analyse gestart voor %s", gebied)

    hoofd = _haal_uitsnede(instellingen.luchtfoto.laag, instellingen.luchtfoto.zoom, gebied, instellingen, toon_voortgang)
    keuze = bepaal_groenbron(
        gebied.bbox,
        hoofdlaag=instellingen.luchtfoto.laag,
        zoom=instellingen.luchtfoto.zoom,
        cache_map=instellingen.cache_map,
        voorkeur=instellingen.luchtfoto.groenlaag,
        minimaal_aandeel=instellingen.luchtfoto.groen_minimaal_aandeel,
    )
    if keuze.laag == hoofd.laag and keuze.zoom == hoofd.zoom:
        groenbeeld = hoofd
    else:
        logger.info("Vegetatie wordt bepaald op %s (zoom %s)", keuze.laag, keuze.zoom)
        groenbeeld = _haal_uitsnede(keuze.laag, keuze.zoom, gebied, instellingen, toon_voortgang)

    registratie = AmsterdamRegistratie(cache_map=instellingen.cache_map).haal_meerdere(
        bronnen_voor_analyse(), gebied.bbox
    )

    analysevlak = _analysevlak(gebied, registratie, instellingen)
    groen = _detecteer_groen(groenbeeld, registratie, instellingen, sam_checkpoint, sam_modeltype)
    verharding = _detecteer_verharding(hoofd, registratie, instellingen)
    detecties = {
        GROEN: _snijd_op(groen, analysevlak),
        VERHARDING: _snijd_op(verharding, analysevlak),
    }
    registratie = {sleutel: _snijd_op(laag, analysevlak) for sleutel, laag in registratie.items()}

    verharding_zonder_groen = _zonder_overlap(detecties[VERHARDING], detecties[GROEN])
    signaleringen = _stel_signaleringen_samen(detecties, verharding_zonder_groen, registratie, instellingen)
    witte_vlekken, dekkingen = _bepaal_vlakdekkendheid(analysevlak, registratie, detecties, instellingen)

    raster = raster_samenvatting(
        pd.concat([signaleringen, witte_vlekken], ignore_index=True) if not witte_vlekken.empty else signaleringen,
        gebied.bbox,
        instellingen.signalering,
    )
    per_thema = samenvatting_per_thema(signaleringen)

    lagen: dict[str, gpd.GeoDataFrame] = {
        "detectie_groen": groen,
        "detectie_verharding": verharding,
        "signaleringen": signaleringen,
        "witte_vlekken": witte_vlekken,
        "signalering_raster": raster,
        "analysevlak": gpd.GeoDataFrame({"naam": ["maaiveld"]}, geometry=[analysevlak], crs=RD),
    }
    for sleutel, laag in registratie.items():
        lagen[f"registratie_{sleutel}"] = laag

    samenvatting = _bouw_samenvatting(
        gebied, hoofd, keuze, groenbeeld, detecties, registratie, signaleringen, witte_vlekken,
        dekkingen, raster, analysevlak, instellingen,
    )
    resultaat = AnalyseResultaat(gebied=gebied, lagen=lagen, samenvatting=samenvatting, per_thema=per_thema)

    if schrijf_bestanden:
        uitvoer_map.mkdir(parents=True, exist_ok=True)
        resultaat.geotiff = schrijf_geotiff(
            uitvoer_map / f"luchtfoto_{hoofd.laag}_zoom{hoofd.zoom}.tif", hoofd.afbeelding, hoofd.transform
        )
        if groenbeeld is not hoofd:
            schrijf_geotiff(
                uitvoer_map / f"luchtfoto_{groenbeeld.laag}_zoom{groenbeeld.zoom}.tif",
                groenbeeld.afbeelding, groenbeeld.transform,
            )
        resultaat.geopackage = schrijf_geopackage(lagen, uitvoer_map / f"{gebied.naam}.gpkg")
        resultaat.geojson_paden = schrijf_geojson(lagen, uitvoer_map / "geojson")
        schrijf_samenvatting(samenvatting, per_thema, uitvoer_map)
    return resultaat


def _haal_uitsnede(laag: str, zoom: int, gebied: Gebied, instellingen: Instellingen, toon_voortgang: bool) -> Uitsnede:
    wmts = LuchtfotoWMTS(
        laag=laag, zoom=zoom, cache_map=instellingen.cache_map, max_werkers=instellingen.luchtfoto.max_werkers
    )
    return wmts.haal_uitsnede(gebied.bbox, toon_voortgang=toon_voortgang)


def _uitsluitmasker(uitsnede: Uitsnede, registratie: dict[str, gpd.GeoDataFrame], buffer_m: float):
    vorm = uitsnede.afbeelding.shape[:2]
    gebouwen = polygonen_naar_masker(registratie["panden"].geometry, vorm, uitsnede.transform, buffer_m=buffer_m)
    water = polygonen_naar_masker(registratie["waterdelen"].geometry, vorm, uitsnede.transform)
    return gebouwen | water


def _detecteer_groen(groenbeeld, registratie, instellingen, sam_checkpoint, sam_modeltype) -> gpd.GeoDataFrame:
    uitsluiten = _uitsluitmasker(groenbeeld, registratie, instellingen.verharding.gebouwbuffer_m)
    masker = bepaal_groenmasker(groenbeeld.afbeelding, groenbeeld.transform, instellingen.groen, uitsluiten)
    groen = detecteer_groen(groenbeeld.afbeelding, groenbeeld.transform, instellingen.groen, masker=masker)
    if sam_checkpoint:
        from luchtfoto_objecten.detectie.sam_contouren import SamContourVerfijner

        verfijner = SamContourVerfijner(checkpoint=sam_checkpoint, modeltype=sam_modeltype)
        groen = verfijner.verfijn(groenbeeld.afbeelding, groenbeeld.transform, groen)
    return groen


def _detecteer_verharding(hoofd, registratie, instellingen) -> gpd.GeoDataFrame:
    uitsluiten = _uitsluitmasker(hoofd, registratie, instellingen.verharding.gebouwbuffer_m)
    return detecteer_verharding(hoofd.afbeelding, hoofd.transform, instellingen.verharding, uitsluitmasker=uitsluiten)


def _zonder_overlap(laag: gpd.GeoDataFrame, aftrek: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Verharding die op de zomerfoto vegetatie blijkt, melden we niet als onbekende verharding."""
    if laag.empty or aftrek.empty:
        return laag
    index = aftrek.sindex
    behouden = []
    for positie, rij in laag.iterrows():
        posities = index.query(rij.geometry, predicate="intersects")
        if len(posities) == 0:
            behouden.append(positie)
            continue
        overlap = unary_union([rij.geometry.intersection(ander) for ander in aftrek.geometry.iloc[posities]])
        if overlap.area / rij.geometry.area < 0.5:
            behouden.append(positie)
    return laag.loc[behouden]


def _stel_signaleringen_samen(detecties, verharding_zonder_groen, registratie, instellingen) -> gpd.GeoDataFrame:
    setregistraties = {
        registratieset: _combineer(
            registratie, tuple(bron for thema in themas_van_set(registratieset) for bron in thema.bronnen)
        )
        for registratieset in REGISTRATIESETS
    }
    delen = []
    for thema in THEMAS:
        detectie = detecties[thema.detectieklasse]
        voor_nieuw = verharding_zonder_groen if thema.detectieklasse == VERHARDING else None
        deel = vergelijk_vlakken(
            detectie,
            _combineer(registratie, thema.bronnen),
            instellingen.signalering,
            thema=thema.naam,
            detectieklasse=thema.detectieklasse,
            detectie_voor_nieuw=voor_nieuw,
            registratieset=setregistraties[thema.registratieset],
        )
        if not deel.empty:
            delen.append(deel)
    if not delen:
        return lege_gdf(SIGNAAL_KOLOMMEN)
    return gpd.GeoDataFrame(pd.concat(delen, ignore_index=True), geometry="geometry", crs=RD)


def _combineer(registratie: dict[str, gpd.GeoDataFrame], bronnen: tuple[str, ...]) -> gpd.GeoDataFrame:
    delen = []
    for bron in bronnen:
        laag = registratie[bron]
        if laag.empty:
            continue
        deel = laag.copy()
        deel["registratiebron"] = bron
        delen.append(deel)
    if not delen:
        return lege_gdf(["registratiebron"])
    return gpd.GeoDataFrame(pd.concat(delen, ignore_index=True), geometry="geometry", crs=RD)


def _analysevlak(gebied: Gebied, registratie: dict[str, gpd.GeoDataFrame], instellingen: Instellingen):
    """Het maaiveld waarover we een uitspraak doen.

    Dat is het gebied zonder panden en zonder water, en standaard nog begrensd tot de
    beheerkaart. Zonder die begrenzing melden binnentuinen en particuliere erven zich als
    ontbrekend groen, terwijl de gemeente die helemaal niet registreert.
    """
    vlak = box(*gebied.bbox)
    for sleutel, buffer_m in (("panden", instellingen.verharding.gebouwbuffer_m), ("waterdelen", 0.0)):
        laag = registratie[sleutel]
        if laag.empty:
            continue
        geometrieen = [geo.buffer(buffer_m) if buffer_m else geo for geo in laag.geometry if geo is not None]
        vlak = vlak.difference(unary_union(geometrieen))

    beheerkaart = registratie.get("beheerkaart")
    if instellingen.signalering.beperk_tot_beheergebied and beheerkaart is not None and not beheerkaart.empty:
        beheergebied = unary_union(list(beheerkaart.geometry)).buffer(0)
        vlak = vlak.intersection(beheergebied)
        logger.info("Analysevlak begrensd tot de beheerkaart: %.0f m2", vlak.area)
    return _alleen_vlakken(vlak)


def _alleen_vlakken(geometrie):
    """Doorsnijden levert soms losse lijnen of punten op, die horen niet in een oppervlaktemaat."""
    if geometrie.geom_type in ("Polygon", "MultiPolygon"):
        return geometrie
    delen = [deel for deel in getattr(geometrie, "geoms", []) if deel.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(delen) if delen else geometrie


def _snijd_op(laag: gpd.GeoDataFrame, vlak) -> gpd.GeoDataFrame:
    if laag.empty or vlak.is_empty:
        return laag
    geknipt = gpd.clip(laag, vlak)
    geknipt = geknipt[~geknipt.geometry.is_empty & geknipt.geometry.notna()].copy()
    geknipt["geometry"] = geknipt.geometry.map(_alleen_vlakken)
    geknipt = geknipt[geknipt.geometry.area > 0]
    return geknipt.reset_index(drop=True)


def _bepaal_vlakdekkendheid(analysevlak, registratie, detecties, instellingen):
    vlekken = []
    dekkingen = {}
    for registratieset in REGISTRATIESETS:
        geometrieen = [
            geo
            for thema in themas_van_set(registratieset)
            for bron in thema.bronnen
            for geo in registratie[bron].geometry
            if geo is not None and not geo.is_empty
        ]
        union = unary_union(geometrieen) if geometrieen else None
        deel, samenvatting = analyseer_vlakdekkendheid(
            analysevlak, union, detecties, instellingen.signalering, thema=registratieset
        )
        dekkingen[registratieset] = samenvatting
        if not deel.empty:
            vlekken.append(deel)
    if not vlekken:
        return lege_gdf(WITTE_VLEK_KOLOMMEN), dekkingen
    return gpd.GeoDataFrame(pd.concat(vlekken, ignore_index=True), geometry="geometry", crs=RD), dekkingen


def _bouw_samenvatting(
    gebied, hoofd, keuze: Bladstandkeuze, groenbeeld, detecties, registratie, signaleringen, witte_vlekken,
    dekkingen, raster, analysevlak, instellingen,
) -> dict:
    signaaltelling = signaleringen["status"].value_counts().to_dict() if not signaleringen.empty else {}
    if not witte_vlekken.empty:
        for status, aantal in witte_vlekken["status"].value_counts().items():
            signaaltelling[status] = signaaltelling.get(status, 0) + int(aantal)
    return {
        "gebied": gebied.naam,
        "omschrijving": gebied.omschrijving,
        "stadsdeel": gebied.stadsdeel,
        "bbox_rd": list(gebied.bbox),
        "oppervlakte_ha": round(gebied.oppervlakte_ha, 2),
        "uitgevoerd_op": datetime.now().isoformat(timespec="seconds"),
        "luchtfoto": {
            "laag": hoofd.laag,
            "zoom": hoofd.zoom,
            "resolutie_m": round(hoofd.resolutie_m, 4),
            "breedte_px": int(hoofd.afbeelding.shape[1]),
            "hoogte_px": int(hoofd.afbeelding.shape[0]),
            "groenlaag": groenbeeld.laag,
            "groenlaag_zoom": groenbeeld.zoom,
            "groenaandeel_pct": round(meet_groenaandeel(groenbeeld.afbeelding) * 100, 1),
            "bladstand_gemeten": keuze.gemeten,
        },
        "maaiveld_m2": round(float(analysevlak.area), 1),
        "detectie": {
            sleutel: {"aantal": int(len(laag)), "oppervlakte_m2": round(float(laag.geometry.area.sum()), 1)}
            for sleutel, laag in detecties.items()
        },
        "registratie": {sleutel: int(len(laag)) for sleutel, laag in registratie.items()},
        "vlakdekkendheid": dekkingen,
        "signalen": {sleutel: int(waarde) for sleutel, waarde in signaaltelling.items()},
        "signalen_hoge_prioriteit": int((signaleringen["prioriteit"] == "hoog").sum()) if not signaleringen.empty else 0,
        "cellen_met_achterstand": int(raster["achterstand_verwacht"].sum()) if not raster.empty else 0,
        "totaal_cellen": int(len(raster)),
        "gebruikte_drempels": {
            "min_dekkingsgraad": instellingen.signalering.min_dekkingsgraad,
            "lage_dekkingsgraad": instellingen.signalering.lage_dekkingsgraad,
            "min_nieuw_oppervlakte_m2": instellingen.signalering.min_nieuw_oppervlakte_m2,
            "min_witte_vlek_m2": instellingen.signalering.min_witte_vlek_m2,
            "cel_signaaldrempel": instellingen.signalering.cel_signaaldrempel,
        },
        "kanttekeningen": [
            "Vegetatie en verharding komen uit verschillende opnamen, want de 8cm-ortho wordt in het "
            "voorjaar zonder blad gevlogen. Het verschil in opnamemoment is zelf ook een bron van afwijkingen.",
            "Onder dichte boomkronen is het maaiveld op een nadirfoto niet te zien, daar telt verharding "
            "als afwijking terwijl de registratie klopt.",
            "Geparkeerde voertuigen, bouwplaatsen en marktkramen bedekken verharding en verlagen de dekkingsgraad.",
            "Objecten aan de rand van het gebied zijn afgeknipt op de bbox, hun oppervlakte is dus lager.",
        ],
    }
