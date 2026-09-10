"""Voorbeeld: loopt de registratie van Amsterdam achter op de luchtfoto?

Je geeft de fysieke grens van een gebied mee, de pijplijn haalt daar de luchtfoto op,
classificeert groen en verharding, en legt die contouren naast de groenobjecten en
verhardingen uit de objectenregistratie openbare ruimte. Wat eruit komt is een lijst met
plekken waar beeld en registratie niet meer op elkaar aansluiten:

    ontbreekt_in_registratie        op de foto zichtbaar, nergens geregistreerd
    niet_zichtbaar_op_luchtfoto     geregistreerd, maar de foto toont er iets anders
    afwijkende_geometrie            het vlak klopt maar voor een deel
    geen_registratie_op_maaiveld    maaiveld dat in geen enkel geregistreerd vlak valt

Draaien vanuit de projectmap:

    python main.py

De classificatie zelf staat in luchtfoto_objecten/detectie/, nu regelgebaseerd op kleur
en textuur. Dat is de plek om andere modellen te proberen: een detector hoeft alleen
dezelfde GeoDataFrame met vlakken terug te geven, dan werkt de vergelijking hieronder
ongewijzigd. detectie/sam_contouren.py laat zien hoe dat eruitziet voor Segment Anything.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from luchtfoto_objecten.gebieden import gebied_uit_bbox
from luchtfoto_objecten.geo_hulp import RD
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.pipeline import voer_analyse_uit
from luchtfoto_objecten.uitvoer import schrijf_geopackage
from luchtfoto_objecten.vergelijking.signalering import BEVESTIGD

# Het analysegebied. Vul GRENS_BESTAND met een GeoJSON, Shapefile of GeoPackage met de
# echte gebiedsgrens, of laat het leeg en pas GRENS_POLYGON aan (RD, EPSG:28992).
GRENS_BESTAND: Path | None = None
GRENS_POLYGON = Polygon(
    [
        (121650, 487150),
        (121950, 487100),
        (122050, 487250),
        (121950, 487450),
        (121700, 487400),
        (121600, 487280),
    ]
)
GEBIEDSNAAM = "voorbeeld_nieuwmarkt"
UITVOER_MAP = Path("data/uitvoer") / GEBIEDSNAAM

# Hoeveel afwijkingen we op het scherm tonen; alles staat altijd in de GeoPackage.
TOON_AANTAL = 25
PRIORITEITSVOLGORDE = {"hoog": 0, "midden": 1, "laag": 2}
AFWIJKING_KOLOMMEN = [
    "thema", "status", "registratiebron", "oppervlakte_m2", "afwijking_m2", "prioriteit", "identificatie",
]


def laad_grens():
    """De gebiedsgrens waarover we een uitspraak doen, in RD (EPSG:28992)."""
    if GRENS_BESTAND is None:
        return GRENS_POLYGON
    grens = gpd.read_file(GRENS_BESTAND)
    if grens.empty:
        raise ValueError(f"{GRENS_BESTAND} bevat geen geometrie")
    if grens.crs is None:
        raise ValueError(f"{GRENS_BESTAND} heeft geen CRS, omzetten naar RD is dan gokwerk")
    return grens.to_crs(RD).union_all()


def knip_op_grens(laag: gpd.GeoDataFrame, grens) -> gpd.GeoDataFrame:
    """De pijplijn rekent op de rechthoek om de grens heen, dit knipt terug naar de grens zelf."""
    if laag.empty:
        return laag
    geknipt = gpd.clip(laag, grens)
    geknipt = geknipt[~geknipt.geometry.is_empty & geknipt.geometry.notna()]
    return geknipt.reset_index(drop=True)


def toon_afwijkingen(signaleringen: gpd.GeoDataFrame, witte_vlekken: gpd.GeoDataFrame) -> None:
    afwijkend = signaleringen[signaleringen["status"] != BEVESTIGD]
    bevestigd = len(signaleringen) - len(afwijkend)
    print(f"\n{bevestigd} geregistreerde vlakken komen overeen met de foto.")

    if afwijkend.empty:
        print("Geen afwijkingen gevonden.")
    else:
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    grens = laad_grens()
    gebied = gebied_uit_bbox(grens.bounds, naam=GEBIEDSNAAM)
    print(f"Analysegebied: {gebied}, grensvlak {grens.area:.0f} m2")

    resultaat = voer_analyse_uit(
        gebied, instellingen=Instellingen.laden(), uitvoer_map=UITVOER_MAP, schrijf_bestanden=False
    )
    lagen = {naam: knip_op_grens(laag, grens) for naam, laag in resultaat.lagen.items()}

    print()
    print(resultaat.toon())
    if not resultaat.per_thema.empty:
        print()
        print(resultaat.per_thema.to_string(index=False))

    toon_afwijkingen(lagen["signaleringen"], lagen["witte_vlekken"])

    UITVOER_MAP.mkdir(parents=True, exist_ok=True)
    pad = schrijf_geopackage(lagen, UITVOER_MAP / f"{GEBIEDSNAAM}.gpkg")
    print(f"\nAlle lagen (te openen in QGIS): {pad}")


if __name__ == "__main__":
    main()
