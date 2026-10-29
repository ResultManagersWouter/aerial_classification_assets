"""Stobbe-analyse: welke stobbes zijn correct, en welke zijn al vervangen door een nieuwe boom?

Draai dit bestand (groene knop in PyCharm, of `python stobbe_main.py`). De uitvoer komt in
output/stobbes_analyse/: een GeoPackage om in QGIS na te lopen, met per stobbe een status,
een advies en links naar het straatbeeld, en een rapport in Markdown. Bij een run over de
hele stad gaat er ook een kopie naar de OneDrive-datamap van het datakwaliteitsdashboard.

De eerste keer duurt het lang (AHN, luchtfoto's, panorama's ophalen); alles wordt bewaard
in "data assets Amsterdam" en daarna hergebruikt. Zie stobbes_analyse.py voor de aanpak.
"""

import logging

from stobbes_analyse import analyseer_stobbes

# Instellingen ---------------------------------------------------------------------------
PROEF_GEBIED = None             # None = hele stad; of een proef in RD, bijvoorbeeld:
# PROEF_GEBIED = (123000, 489000, 124500, 490500)   # een stuk Noord, een paar minuten
OPNIEUW_METEN = False           # True = AHN en luchtfoto opnieuw meten in plaats van uit de opslag
STRAATBEELD_OPHALEN = True      # panorama per stobbe opzoeken; False = alleen Google-links
KOPIE_NAAR_ONEDRIVE = True      # uitvoer ook in de datamap van het datakwaliteitsdashboard
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    # Een AHN-tegel die niet bestaat (alleen water) geeft een 403; dat is verwacht en afgevangen
    logging.getLogger("rasterio._env").setLevel(logging.WARNING)
    analyseer_stobbes(bbox=PROEF_GEBIED, opnieuw=OPNIEUW_METEN, straatbeeld_ophalen=STRAATBEELD_OPHALEN,
                      kopie_naar_onedrive=KOPIE_NAAR_ONEDRIVE)
