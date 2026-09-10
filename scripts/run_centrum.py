"""Analyse van een blok in Amsterdam Centrum, bedoeld om vanuit PyCharm te draaien."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from luchtfoto_objecten.gebieden import gebied_op_naam
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.pipeline import voer_analyse_uit

GEBIED = "centrum_plantage"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    resultaat = voer_analyse_uit(gebied_op_naam(GEBIED), Instellingen.laden())

    print()
    print(resultaat.toon())
    print()
    if not resultaat.per_thema.empty:
        print(resultaat.per_thema.to_string(index=False))
    print(f"\nOpen in QGIS: {resultaat.geopackage}")


if __name__ == "__main__":
    main()
