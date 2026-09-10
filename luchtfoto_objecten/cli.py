from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from luchtfoto_objecten.gebieden import Gebied, gebied_op_naam, gebied_uit_bbox, laad_gebieden
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.pipeline import voer_analyse_uit
from luchtfoto_objecten.raster import schrijf_geotiff
from luchtfoto_objecten.referentie.amsterdam import BRONNEN, AmsterdamRegistratie
from luchtfoto_objecten.themas import THEMAS, bronnen_voor_analyse
from luchtfoto_objecten.uitvoer import schrijf_geopackage
from luchtfoto_objecten.wmts import LuchtfotoWMTS, haal_beschikbare_lagen, jaarlagen_nieuwste_eerst, tegelbereik


def bouw_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="luchtfoto-objecten",
        description="Vlakken herkennen op de 8cm luchtfoto en vergelijken met de registratie van Amsterdam.",
    )
    parser.add_argument("--verbose", action="store_true", help="toon uitgebreide logging")
    subparsers = parser.add_subparsers(dest="commando", required=True)

    subparsers.add_parser("gebieden", help="toon de beschikbare analysegebieden")
    subparsers.add_parser("lagen", help="toon de luchtfotolagen bij PDOK en de registraties van Amsterdam")

    for naam, hulp in [("analyse", "volledige analyse met vergelijking"), ("download", "alleen de luchtfoto ophalen")]:
        deelparser = subparsers.add_parser(naam, help=hulp)
        _voeg_gebiedsargumenten_toe(deelparser)
        deelparser.add_argument("--laag", help="luchtfotolaag, standaard Actueel_orthoHR uit config")
        deelparser.add_argument("--zoom", type=int, help="zoomniveau, 15 is 10,5 cm/px en 16 is 5,25 cm/px")
        deelparser.add_argument("--uitvoer", type=Path, help="map voor de resultaten")
        if naam == "analyse":
            deelparser.add_argument("--groenlaag", help="auto, gelijk, of een expliciete laag voor de vegetatie")
            deelparser.add_argument("--sam-checkpoint", help="pad naar een SAM-checkpoint voor scherpere contouren")
            deelparser.add_argument("--sam-modeltype", default="vit_b", help="vit_b, vit_l of vit_h")

    registratieparser = subparsers.add_parser("registratie", help="alleen de gemeentelijke registratie ophalen")
    _voeg_gebiedsargumenten_toe(registratieparser)
    registratieparser.add_argument("--bronnen", nargs="+", choices=sorted(BRONNEN), default=bronnen_voor_analyse())
    registratieparser.add_argument("--uitvoer", type=Path, help="map voor de resultaten")
    return parser


def _voeg_gebiedsargumenten_toe(parser: argparse.ArgumentParser) -> None:
    groep = parser.add_mutually_exclusive_group(required=True)
    groep.add_argument("--gebied", help="naam uit config/gebieden.yaml")
    groep.add_argument(
        "--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="eigen gebied in Rijksdriehoek",
    )


def _bepaal_gebied(argumenten) -> Gebied:
    if argumenten.gebied:
        return gebied_op_naam(argumenten.gebied)
    return gebied_uit_bbox(tuple(argumenten.bbox))


def _instellingen_met_overschrijving(argumenten) -> Instellingen:
    instellingen = Instellingen.laden()
    if getattr(argumenten, "laag", None):
        instellingen.luchtfoto.laag = argumenten.laag
    if getattr(argumenten, "zoom", None):
        instellingen.luchtfoto.zoom = argumenten.zoom
    if getattr(argumenten, "groenlaag", None):
        instellingen.luchtfoto.groenlaag = argumenten.groenlaag
    return instellingen


def _toon_gebieden() -> None:
    for naam, gebied in sorted(laad_gebieden().items()):
        print(f"{naam:28} {gebied.breedte_m:5.0f} x {gebied.hoogte_m:5.0f} m  {gebied.oppervlakte_ha:6.1f} ha")
        if gebied.omschrijving:
            print(f"{'':28} {gebied.omschrijving}")


def _toon_lagen() -> None:
    instellingen = Instellingen.laden()
    lagen = haal_beschikbare_lagen(instellingen.cache_map)
    print("Luchtfotolagen van de landelijke voorziening Beeldmateriaal (PDOK):")
    for identificatie in ["Actueel_orthoHR", "Actueel_ortho25", *jaarlagen_nieuwste_eerst(lagen)]:
        if identificatie in lagen:
            print(f"  {identificatie:22} {lagen[identificatie]}")
    print("\nRegistraties van de gemeente Amsterdam:")
    for sleutel, bron in BRONNEN.items():
        print(f"  {sleutel:24} {bron.omschrijving}")
    print("\nThema's in de vergelijking:")
    for thema in THEMAS:
        print(f"  {thema.naam:24} {thema.registratieset:18} tegen detectie '{thema.detectieklasse}'")


def _download(argumenten) -> None:
    gebied = _bepaal_gebied(argumenten)
    instellingen = _instellingen_met_overschrijving(argumenten)
    kolom_min, rij_min, kolom_max, rij_max = tegelbereik(gebied.bbox, instellingen.luchtfoto.zoom)
    aantal = (kolom_max - kolom_min + 1) * (rij_max - rij_min + 1)
    print(f"{gebied}, {aantal} tegels op zoom {instellingen.luchtfoto.zoom}")

    wmts = LuchtfotoWMTS(
        laag=instellingen.luchtfoto.laag,
        zoom=instellingen.luchtfoto.zoom,
        cache_map=instellingen.cache_map,
        max_werkers=instellingen.luchtfoto.max_werkers,
    )
    uitsnede = wmts.haal_uitsnede(gebied.bbox)
    uitvoer_map = argumenten.uitvoer or (instellingen.uitvoer_map / gebied.naam)
    pad = schrijf_geotiff(
        uitvoer_map / f"luchtfoto_{uitsnede.laag}_zoom{uitsnede.zoom}.tif", uitsnede.afbeelding, uitsnede.transform
    )
    print(f"Weggeschreven: {pad}")


def _registratie(argumenten) -> None:
    gebied = _bepaal_gebied(argumenten)
    instellingen = Instellingen.laden()
    klant = AmsterdamRegistratie(cache_map=instellingen.cache_map)
    lagen = {f"registratie_{sleutel}": klant.haal(sleutel, gebied.bbox) for sleutel in argumenten.bronnen}
    for naam, laag in lagen.items():
        oppervlak = laag.geometry.area.sum() if not laag.empty else 0.0
        print(f"{naam:36} {len(laag):6} objecten  {oppervlak:10.0f} m2")
    uitvoer_map = argumenten.uitvoer or (instellingen.uitvoer_map / gebied.naam)
    pad = schrijf_geopackage(lagen, uitvoer_map / f"{gebied.naam}_registratie.gpkg")
    print(f"Weggeschreven: {pad}")


def _analyse(argumenten) -> None:
    gebied = _bepaal_gebied(argumenten)
    instellingen = _instellingen_met_overschrijving(argumenten)
    resultaat = voer_analyse_uit(
        gebied,
        instellingen=instellingen,
        uitvoer_map=argumenten.uitvoer,
        sam_checkpoint=argumenten.sam_checkpoint,
        sam_modeltype=argumenten.sam_modeltype,
    )
    print()
    print(resultaat.toon())
    print()
    if not resultaat.per_thema.empty:
        print(resultaat.per_thema.to_string(index=False))
    print(f"\nGeoPackage: {resultaat.geopackage}")


def main(argumentenlijst: list[str] | None = None) -> int:
    parser = bouw_parser()
    argumenten = parser.parse_args(argumentenlijst)
    logging.basicConfig(
        level=logging.DEBUG if argumenten.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if argumenten.commando == "gebieden":
        _toon_gebieden()
    elif argumenten.commando == "lagen":
        _toon_lagen()
    elif argumenten.commando == "download":
        _download(argumenten)
    elif argumenten.commando == "registratie":
        _registratie(argumenten)
    elif argumenten.commando == "analyse":
        _analyse(argumenten)
    return 0


if __name__ == "__main__":
    sys.exit(main())
