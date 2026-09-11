"""Loopt de registratie van Amsterdam achter op de luchtfoto?

Je geeft een gebied mee, de pijplijn haalt daar de luchtfoto op, classificeert groen en
verharding, en legt die contouren naast de groenobjecten en verhardingen uit de
objectenregistratie openbare ruimte. Wat eruit komt is een lijst met plekken waar beeld
en registratie niet meer op elkaar aansluiten:

    ontbreekt_in_registratie        op de foto zichtbaar, nergens geregistreerd
    niet_zichtbaar_op_luchtfoto     geregistreerd, maar de foto toont er iets anders
    afwijkende_geometrie            het vlak klopt maar voor een deel
    geen_registratie_op_maaiveld    maaiveld dat in geen enkel geregistreerd vlak valt

Draaien, met het voorbeeldgebied in Centrum:

    python main.py

Een eigen gebied, in Rijksdriehoek (EPSG:28992). QGIS toont een extent als
xmin, xmax, ymin, ymax, dus die volgorde kan zo overgenomen worden:

    python main.py --extent 121641.1653 122215.9413 486408.2909 487051.9774 --naam centrum

Of in de volgorde die de rest van dit project gebruikt, of vanuit een bestand met de
echte gebiedsgrens (GeoJSON, Shapefile, GeoPackage; elke polygon mag, niet alleen een
rechthoek):

    python main.py --bbox 121641 486408 122215 487051 --naam centrum
    python main.py --grens data/aoi/buurt.geojson --naam buurt

Met --modellen wordt er ook een modelvergelijking gedraaid: de huidige drempels naast
logistische regressie, random forest en gradient boosting. Dat vraagt scikit-learn,
zie requirements-ml.txt.

Alles komt terecht in output/<naam>/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from luchtfoto_objecten.gebieden import gebied_op_naam, gebied_uit_bbox
from luchtfoto_objecten.geo_hulp import RD
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.pipeline import voer_analyse_uit
from luchtfoto_objecten.uitvoer import schrijf_geopackage
from luchtfoto_objecten.vergelijking.signalering import BEVESTIGD

# Voorbeeldgebied in Amsterdam Centrum, gebruikt als je geen gebied opgeeft.
VOORBEELD_EXTENT = (121641.1653, 122215.9413, 486408.2909, 487051.9774)

# De BGT-terreindelen zitten wel in de vergelijking, maar niet in de uitvoer: ze leggen de
# hele kaart dicht en maken het lastig om de detectie tegen de luchtfoto te bekijken.
LAGEN_NIET_SCHRIJVEN = ("registratie_begroeideterreindelen", "registratie_onbegroeideterreindelen")

TOON_AANTAL = 25
PRIORITEITSVOLGORDE = {"hoog": 0, "midden": 1, "laag": 2}
AFWIJKING_KOLOMMEN = [
    "thema", "status", "registratiebron", "oppervlakte_m2", "afwijking_m2", "prioriteit", "identificatie",
]


def bouw_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vergelijk de registratie van Amsterdam met de luchtfoto voor één gebied."
    )
    gebiedsgroep = parser.add_mutually_exclusive_group()
    gebiedsgroep.add_argument(
        "--extent", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="gebied in RD, in de volgorde die QGIS toont",
    )
    gebiedsgroep.add_argument(
        "--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="gebied in RD, in de volgorde van dit project",
    )
    gebiedsgroep.add_argument("--grens", type=Path, help="bestand met de gebiedsgrens als polygon")
    gebiedsgroep.add_argument("--gebied", help="naam uit config/gebieden.yaml, bijvoorbeeld noord_vliegenbos")
    parser.add_argument("--naam", help="naam van de map onder output/, standaard de gebiedsnaam")
    parser.add_argument("--zoom", type=int, help="15 is 10,5 cm/px, 16 is 5,25 cm/px en viermaal zoveel tegels")
    parser.add_argument(
        "--modellen", action="store_true",
        help="zet per model een detectielaag in de GeoPackage en scoor ze tegen de BGT",
    )
    parser.add_argument(
        "--bomen", action="store_true",
        help="onderzoek of het gevonden groen een boomkroon is of echt groen maaiveld",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="tien parameterinstellingen per klasse, in aparte GeoPackages om te beoordelen",
    )
    parser.add_argument(
        "--jaren", type=int, metavar="N", nargs="?", const=4,
        help="vergelijk groen en verharding over de laatste N jaargangen, standaard 4",
    )
    return parser


def bepaal_grens(argumenten):
    """De gebiedsgrens waarover we een uitspraak doen, in RD (EPSG:28992)."""
    if argumenten.gebied:
        return gebied_op_naam(argumenten.gebied).als_polygon()
    if argumenten.grens:
        grens = gpd.read_file(argumenten.grens)
        if grens.empty:
            raise SystemExit(f"{argumenten.grens} bevat geen geometrie")
        if grens.crs is None:
            raise SystemExit(f"{argumenten.grens} heeft geen CRS, omzetten naar RD is dan gokwerk")
        return grens.to_crs(RD).union_all()
    if argumenten.bbox:
        return box(*argumenten.bbox)
    xmin, xmax, ymin, ymax = argumenten.extent or VOORBEELD_EXTENT
    return box(xmin, ymin, xmax, ymax)


def knip_op_grens(laag: gpd.GeoDataFrame, grens) -> gpd.GeoDataFrame:
    """De pijplijn rekent op de rechthoek om de grens heen, dit knipt terug naar de grens zelf."""
    if laag.empty:
        return laag
    geknipt = gpd.clip(laag, grens)
    geknipt = geknipt[~geknipt.geometry.is_empty & geknipt.geometry.notna()]
    return geknipt.reset_index(drop=True)


def toon_afwijkingen(signaleringen: gpd.GeoDataFrame, witte_vlekken: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    afwijkend = signaleringen[signaleringen["status"] != BEVESTIGD]
    print(f"\n{len(signaleringen) - len(afwijkend)} geregistreerde vlakken komen overeen met de foto.")

    if afwijkend.empty:
        print("Geen afwijkingen gevonden.")
        return afwijkend

    print(f"\n{len(afwijkend)} vlakken wijken af, per status:")
    for status, aantal in afwijkend["status"].value_counts().items():
        oppervlak = afwijkend.loc[afwijkend["status"] == status, "afwijking_m2"].sum()
        print(f"  {status:32} {aantal:4}  samen {oppervlak:9.0f} m2")

    gesorteerd = afwijkend.assign(_volgorde=afwijkend["prioriteit"].map(PRIORITEITSVOLGORDE)).sort_values(
        ["_volgorde", "afwijking_m2"], ascending=[True, False]
    )
    print(f"\nGrootste afwijkingen (top {TOON_AANTAL}):")
    print(gesorteerd.head(TOON_AANTAL)[AFWIJKING_KOLOMMEN].to_string(index=False))

    if not witte_vlekken.empty:
        print(
            f"\n{len(witte_vlekken)} stukken maaiveld vallen in geen enkel geregistreerd vlak, "
            f"samen {witte_vlekken['oppervlakte_m2'].sum():.0f} m2."
        )
    return gesorteerd.drop(columns="_volgorde")


def _toon_boomtoets(lagen, gebied, instellingen) -> None:
    """Toetst de kroon- en groenveldlaag tegen het bomenregister en de groenobjecten."""
    from luchtfoto_objecten.groenstructuur import toets_tegen_register
    from luchtfoto_objecten.referentie.amsterdam import AmsterdamRegistratie

    structuur = {naam: laag for naam, laag in lagen.items() if naam.startswith("detectie_")}
    if not structuur:
        return
    klant = AmsterdamRegistratie(cache_map=instellingen.cache_map)
    toets = toets_tegen_register(
        {"detectie_boomkroon": lagen.get("detectie_boomkroon_ir", lagen.get("detectie_boomkroon")),
         "detectie_groenstrook": lagen.get("detectie_groenveld_ir", lagen.get("detectie_groenstrook"))},
        klant.haal("bomen", gebied.bbox),
        klant.haal("groenobjecten", gebied.bbox),
    )
    if not toets.empty:
        print("\nToets tegen de registratie:")
        print(toets.to_string(index=False))


def _voeg_beelden_toe(pad: Path, gebied, instellingen) -> None:
    """Zet de opnamen waarop de classificatie rust als rasterlaag in de GeoPackage.

    Zo open je één bestand in QGIS en heb je de foto en de vlakken bij elkaar, inclusief
    het infrarood waarop vegetatie het duidelijkst van verharding te onderscheiden is.
    """
    from luchtfoto_objecten.infrarood import haal_infrarood
    from luchtfoto_objecten.modeldetectie import haal_beelden
    from luchtfoto_objecten.raster import schrijf_raster_in_geopackage

    hoofd, groenbeeld, _, _ = haal_beelden(gebied, instellingen)
    beelden = [hoofd]
    if groenbeeld.laag != hoofd.laag:
        beelden.append(groenbeeld)
    infrarood = haal_infrarood(gebied, instellingen)
    if infrarood is not None:
        beelden.append(infrarood)
    for uitsnede in beelden:
        schrijf_raster_in_geopackage(pad, f"luchtfoto_{uitsnede.laag}", uitsnede.afbeelding, uitsnede.transform)
    print("Beelden in de GeoPackage: " + ", ".join(f"luchtfoto_{u.laag}" for u in beelden))


def main(argumentenlijst: list[str] | None = None) -> None:
    argumenten = bouw_parser().parse_args(argumentenlijst)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    instellingen = Instellingen.laden()
    if argumenten.zoom:
        instellingen.luchtfoto.zoom = argumenten.zoom

    grens = bepaal_grens(argumenten)
    naam = argumenten.naam or argumenten.gebied or "centrum_vergelijking"
    gebied = gebied_uit_bbox(grens.bounds, naam=naam)
    uitvoer_map = Path("output") / naam
    uitvoer_map.mkdir(parents=True, exist_ok=True)
    print(f"Analysegebied: {gebied}, grensvlak {grens.area:.0f} m2")

    if argumenten.jaren:
        from luchtfoto_objecten.meerjaren import analyseer as analyseer_jaren

        lagen, overzicht, veranderingen, assets = analyseer_jaren(gebied, instellingen, argumenten.jaren)
        if overzicht.empty:
            raise SystemExit("Geen bruikbare jaargangen gevonden voor dit gebied")
        print("\nPer jaargang:")
        print(overzicht.to_string(index=False))
        if not veranderingen.empty:
            print("\nVerschil tussen opeenvolgende jaren:")
            print(veranderingen.to_string(index=False))
        pad = schrijf_geopackage({naam: knip_op_grens(laag, grens) for naam, laag in lagen.items()},
                                 uitvoer_map / "meerjaren.gpkg")
        overzicht.to_csv(uitvoer_map / "meerjaren_overzicht.csv", index=False)
        veranderingen.to_csv(uitvoer_map / "meerjaren_verschillen.csv", index=False)
        if not assets.empty:
            assets.to_csv(uitvoer_map / "meerjaren_per_asset.csv", index=False)
            print(f"\n{len(assets)} geregistreerde objecten met een dekking per jaar in meerjaren_per_asset.csv")
        print(f"\nLagen per jaar: {pad}")
        return

    if argumenten.sweep:
        from luchtfoto_objecten.sweep import voer_sweep_uit

        for naam, tabel in voer_sweep_uit(gebied, instellingen, uitvoer_map).items():
            print(f"\n{naam}.gpkg:")
            print(tabel.to_string(index=False))
        print(f"\nDrie GeoPackages in {uitvoer_map}/, open ze los in QGIS.")
        return

    resultaat = voer_analyse_uit(
        gebied, instellingen=instellingen, uitvoer_map=uitvoer_map, schrijf_bestanden=False
    )
    lagen = {
        laagnaam: knip_op_grens(laag, grens)
        for laagnaam, laag in resultaat.lagen.items()
        if laagnaam not in LAGEN_NIET_SCHRIJVEN
    }

    print()
    print(resultaat.toon())
    if not resultaat.per_thema.empty:
        print()
        print(resultaat.per_thema.to_string(index=False))

    afwijkend = toon_afwijkingen(lagen["signaleringen"], lagen["witte_vlekken"])

    if argumenten.modellen:
        from luchtfoto_objecten.modeldetectie import bouw_detectielagen
        from luchtfoto_objecten.modelvergelijking import vergelijk_modellen

        modellagen, overzicht = bouw_detectielagen(gebied, instellingen)
        lagen.update({laagnaam: knip_op_grens(laag, grens) for laagnaam, laag in modellagen.items()})
        print("\nWat elk model detecteert, als eigen laag in de GeoPackage:")
        print(overzicht.to_string(index=False))
        overzicht.to_csv(uitvoer_map / "modeloverzicht.csv", index=False)

        vergelijking, belang = vergelijk_modellen(gebied, instellingen)
        print("\nModelvergelijking, referentie is de BGT, west getraind en oost gescoord:")
        print(vergelijking.to_string(index=False))
        vergelijking.to_csv(uitvoer_map / "modelvergelijking.csv", index=False)
        belang.to_csv(uitvoer_map / "kenmerkbelang.csv", index=False)

    if argumenten.bomen:
        from luchtfoto_objecten.boomherkenning import analyseer

        vlakken, medianen, scores = analyseer(gebied, instellingen)
        lagen["groen_boom_of_vlak"] = knip_op_grens(vlakken, grens)
        _toon_boomtoets(lagen, gebied, instellingen)
        print("\nBoomkroon of groenvlak, geijkt op het bomenregister:")
        print(vlakken["label"].value_counts(dropna=False).to_string())
        print("\nMediaan per kenmerk:")
        print(medianen.to_string(index=False))
        if not scores.empty:
            print("\nTe scheiden op beeldkenmerken alleen, met kruisvalidatie:")
            print(scores.to_string(index=False))
            scores.to_csv(uitvoer_map / "boomherkenning_score.csv", index=False)
        vlakken.drop(columns="geometry").to_csv(uitvoer_map / "boom_of_vlak.csv", index=False)

    pad = schrijf_geopackage(lagen, uitvoer_map / f"{naam}.gpkg")
    _voeg_beelden_toe(pad, gebied, instellingen)
    resultaat.per_thema.to_csv(uitvoer_map / "samenvatting_per_thema.csv", index=False)
    if not afwijkend.empty:
        afwijkend.drop(columns="geometry").to_csv(uitvoer_map / "afwijkingen.csv", index=False)
    print(f"\nAlle lagen (te openen in QGIS): {pad}")

    print(f"\nUitvoer staat in {uitvoer_map}/")


if __name__ == "__main__":
    main()
