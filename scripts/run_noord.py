"""Contouren intekenen in Amsterdam Noord, zonder de vergelijking met de registratie.

Noord dient hier als proeftuin voor de detectie zelf. De vlakken komen als GeoPackage
naar buiten zodat je de contouren in QGIS kunt nalopen en de drempels kunt bijstellen.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from luchtfoto_objecten.bladstand import bepaal_groenbron
from luchtfoto_objecten.detectie.groen import detecteer_groen
from luchtfoto_objecten.detectie.verharding import detecteer_verharding
from luchtfoto_objecten.gebieden import gebied_op_naam
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.raster import polygonen_naar_masker, schrijf_geotiff
from luchtfoto_objecten.referentie.amsterdam import AmsterdamRegistratie
from luchtfoto_objecten.uitvoer import schrijf_geojson, schrijf_geopackage
from luchtfoto_objecten.wmts import LuchtfotoWMTS

GEBIEDEN = ["noord_vliegenbos", "noord_ndsm"]


def teken_contouren(naam: str, instellingen: Instellingen) -> None:
    gebied = gebied_op_naam(naam)
    registratie = AmsterdamRegistratie(cache_map=instellingen.cache_map)
    panden = registratie.haal("panden", gebied.bbox)
    waterdelen = registratie.haal("waterdelen", gebied.bbox)

    def uitsnede_van(laag: str, zoom: int):
        return LuchtfotoWMTS(
            laag=laag, zoom=zoom, cache_map=instellingen.cache_map,
            max_werkers=instellingen.luchtfoto.max_werkers,
        ).haal_uitsnede(gebied.bbox)

    def maskers(uitsnede):
        vorm = uitsnede.afbeelding.shape[:2]
        gebouwen = polygonen_naar_masker(panden.geometry, vorm, uitsnede.transform, buffer_m=0.5)
        return gebouwen | polygonen_naar_masker(waterdelen.geometry, vorm, uitsnede.transform)

    hoofd = uitsnede_van(instellingen.luchtfoto.laag, instellingen.luchtfoto.zoom)
    keuze = bepaal_groenbron(
        gebied.bbox, instellingen.luchtfoto.laag, instellingen.luchtfoto.zoom, instellingen.cache_map,
        voorkeur=instellingen.luchtfoto.groenlaag, minimaal_aandeel=instellingen.luchtfoto.groen_minimaal_aandeel,
    )
    groenbeeld = hoofd if keuze.laag == hoofd.laag and keuze.zoom == hoofd.zoom else uitsnede_van(keuze.laag, keuze.zoom)

    groen = detecteer_groen(groenbeeld.afbeelding, groenbeeld.transform, instellingen.groen, maskers(groenbeeld))
    verharding = detecteer_verharding(hoofd.afbeelding, hoofd.transform, instellingen.verharding, maskers(hoofd))

    lagen = {"detectie_groen": groen, "detectie_verharding": verharding}
    uitvoer_map = instellingen.uitvoer_map / gebied.naam
    schrijf_geotiff(uitvoer_map / f"luchtfoto_{hoofd.laag}_zoom{hoofd.zoom}.tif", hoofd.afbeelding, hoofd.transform)
    pad = schrijf_geopackage(lagen, uitvoer_map / f"{gebied.naam}_contouren.gpkg")
    schrijf_geojson(lagen, uitvoer_map / "geojson")

    print(f"{gebied.naam}: groen {len(groen)} vlakken / {groen.geometry.area.sum():.0f} m2, "
          f"verharding {len(verharding)} vlakken / {verharding.geometry.area.sum():.0f} m2 -> {pad}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    instellingen = Instellingen.laden()
    for naam in GEBIEDEN:
        teken_contouren(naam, instellingen)


if __name__ == "__main__":
    main()
