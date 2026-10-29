"""Bomen-analyse: staat elke geregistreerde boom er echt, en maar één keer?

Draai dit bestand (groene knop in PyCharm, of `python bomen_main.py`). De uitvoer komt in
output/bomen_analyse/: een GeoPackage met de lagen solitaire_bomen, alle_bomen,
te_controleren en dubbele_punten, en een rapport in Markdown. Bij een run over de hele stad
gaat er ook een kopie naar de OneDrive-datamap van het datakwaliteitsdashboard.

De eerste run over de hele stad duurt een paar uur: AHN-kronen voor de hele stad en twee
luchtfotojaargangen voor ruim 300.000 bomen. Alles wordt bewaard in "data assets Amsterdam"
en daarna hergebruikt. Zie bomen_analyse.py voor de aanpak.
"""

import logging

from bomen_analyse import analyseer_bomen

# Instellingen ---------------------------------------------------------------------------
PROEF_GEBIED = None             # None = hele stad; of een proef in RD, bijvoorbeeld:
# PROEF_GEBIED = (121000, 486000, 122000, 487000)   # een kilometer in Zuid
OPNIEUW_METEN = False           # True = luchtfoto opnieuw meten in plaats van uit de opslag
STRAATBEELD_OPHALEN = True      # panorama opzoeken bij de bomen die te controleren zijn
KOPIE_NAAR_ONEDRIVE = True      # uitvoer ook in de datamap van het datakwaliteitsdashboard
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("rasterio._env").setLevel(logging.WARNING)
    analyseer_bomen(bbox=PROEF_GEBIED, opnieuw=OPNIEUW_METEN, straatbeeld_ophalen=STRAATBEELD_OPHALEN,
                    kopie_naar_onedrive=KOPIE_NAAR_ONEDRIVE)
