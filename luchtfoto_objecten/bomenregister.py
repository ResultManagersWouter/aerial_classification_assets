"""De bomenregistratie van Amsterdam uit de bomen-WFS, met hergebruik via de assetmap.

    https://api.data.amsterdam.nl/v1/wfs/bomen

Twee lagen zijn hier van belang. stamgegevens heeft een punt per boom, ook voor stobbes
(type_object = Stobbe), en kapenherplant heeft per kap- en herplanttraject de kapdatum, de
herplantinspectie en het planten van de nieuwe boom. Een volledige export van stamgegevens
duurt anderhalve minuut; die wordt een week hergebruikt uit de assetmap.
"""

from __future__ import annotations

import logging
import time

import geopandas as gpd
import pandas as pd
import requests

from luchtfoto_objecten import opslag

logger = logging.getLogger(__name__)

WFS_BOMEN = "https://api.data.amsterdam.nl/v1/wfs/bomen"
WFS_PAGINA = 5000

# Standplaatsen waar een boom los in de verharding staat: daar is hij van boven en vanaf
# de straat goed te zien. In gras en plantsoen gaat het meestal ook, in bos niet.
VERHARD = {"Elementenverharding", "Asfaltverharding", "Verhardingsobject", "Betonverharding", "Halfverharding",
           "Onverhard", "Kunstgras"}
BOS = {"Bosplantsoen"}

KAP_DATUMS = ("datum_herplantinspectie", "kapmaatregel_datum_uitgevoerd", "plantmaatregel_datum_uitgevoerd",
              "groeiplaatsmaatregel_datum_uitgevoerd", "planten_datum_toezicht", "kap_datum_toezicht")


def filter_gelijk(veld: str, waarde: str) -> str:
    """FES 2.0-filter veld = waarde, voor de FILTER-parameter van de WFS."""
    return (
        '<fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0"><fes:PropertyIsEqualTo>'
        f"<fes:ValueReference>{veld}</fes:ValueReference><fes:Literal>{waarde}</fes:Literal>"
        "</fes:PropertyIsEqualTo></fes:Filter>"
    )


def wfs_ophalen(laag: str, filter_xml: str | None = None) -> gpd.GeoDataFrame:
    """Alle features van een laag uit de bomen-WFS, per pagina van 5000."""
    basis = {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
             "TYPENAMES": f"app:{laag}", "SRSNAME": "EPSG:28992", "OUTPUTFORMAT": "geojson"}
    if filter_xml:
        basis["FILTER"] = filter_xml
    delen, start = [], 0
    while True:
        for poging in range(1, 6):
            try:
                antwoord = requests.get(WFS_BOMEN, params={**basis, "COUNT": WFS_PAGINA, "STARTINDEX": start},
                                        timeout=300)
                antwoord.raise_for_status()
                break
            except requests.RequestException as fout:
                if poging == 5:
                    raise
                logger.warning("WFS-fout (%s), opnieuw over %ss", fout, 5 * poging)
                time.sleep(5 * poging)
        features = antwoord.json()["features"]
        if features:
            delen.append(gpd.GeoDataFrame.from_features(features, crs="EPSG:28992"))
        if len(features) < WFS_PAGINA:
            break
        start += WFS_PAGINA
    gdf = pd.concat(delen, ignore_index=True)
    # De GeoJSON van de WFS zet het id ook in 'name'
    if "name" in gdf and gdf["name"].equals(gdf["id"]):
        gdf = gdf.drop(columns="name")
    logger.info("WFS %s: %s features", laag, f"{len(gdf):,}")
    return gdf


def alle_boompunten(max_leeftijd_dagen: int = 7) -> gpd.GeoDataFrame:
    """Alle boompunten, stobbes inbegrepen, uit de opslag of vers uit de WFS."""
    gdf = opslag.wfs_export("stamgegevens", lambda: wfs_ophalen("stamgegevens"), max_leeftijd_dagen)
    zonder_plek = gdf.geometry.isna() | gdf.geometry.is_empty
    if zonder_plek.any():
        # Een paar boompunten in de WFS hebben geen geometrie; daar valt niets aan te meten
        logger.warning("%s boompunten zonder geometrie overgeslagen (id's: %s)", int(zonder_plek.sum()),
                       ", ".join(map(str, gdf.loc[zonder_plek, "id"].head(10))))
        gdf = gdf[~zonder_plek].reset_index(drop=True)
    gdf["mutatie_datum"] = pd.to_datetime(gdf["mutatie_datum"], errors="coerce", utc=True)
    gdf["geldig_vanaf"] = pd.to_datetime(gdf["geldig_vanaf"], errors="coerce")
    return gdf


def kap_en_herplant(max_leeftijd_dagen: int = 7) -> pd.DataFrame:
    """Kap- en herplanttrajecten, met boom_id als geheel getal en de datums als datum."""
    kap = opslag.wfs_export("kapenherplant", lambda: wfs_ophalen("kapenherplant"), max_leeftijd_dagen)
    kap = pd.DataFrame(kap.drop(columns="geometry"))
    for kolom in ("boom_id", "boom_nieuw_id"):
        kap[kolom] = pd.to_numeric(kap[kolom], errors="coerce").astype("Int64")
    for kolom in KAP_DATUMS:
        kap[kolom] = pd.to_datetime(kap[kolom], errors="coerce")
    return kap


def splits(boompunten: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Stobbes (met het filter type_object = Stobbe) en levende bomen (de rest)."""
    is_stobbe = boompunten["type_object"].eq("Stobbe")
    return boompunten[is_stobbe].reset_index(drop=True), boompunten[~is_stobbe].reset_index(drop=True)


def omgeving(bomen: gpd.GeoDataFrame, alle_punten: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    """Waar staat een boom: straat, groen of bos, en hoeveel geregistreerde buren dichtbij.

    `alle_punten` zijn de punten waartegen de buren geteld worden, standaard de bomen zelf.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    alle_punten = bomen if alle_punten is None else alle_punten
    boom = cKDTree(np.c_[alle_punten.geometry.x, alle_punten.geometry.y])
    xy = np.c_[bomen.geometry.x, bomen.geometry.y]
    zelf_erin = alle_punten is bomen
    afstand, _ = boom.query(xy, k=6)
    afstand = afstand[:, 1:] if zelf_erin else afstand[:, :-1]
    buur = afstand[:, 0]
    # Voor 'losstaand' tellen punten binnen 2,5 m niet mee: dat zijn mogelijk dubbele
    # registraties van dezelfde boom, en die wil je juist kunnen vinden
    verder = np.where(afstand > 2.5, afstand, np.inf).min(axis=1)
    binnen_5 = np.array([len(i) for i in boom.query_ball_point(xy, 5.0)]) - (1 if zelf_erin else 0)
    standplaats = bomen["standplaats"].fillna("")
    soort = np.where(standplaats.isin(VERHARD), "straat", np.where(standplaats.isin(BOS), "bos", "groen"))
    soort = np.where((binnen_5 >= 3) & (soort != "straat"), "bos", soort)
    return pd.DataFrame({
        "omgeving": soort,
        "buurboom_afstand_m": np.round(buur, 2),
        "bomen_binnen_5m": binnen_5,
        # Losstaand: aan de straat of in het groen, niet in bos, en de eerstvolgende andere
        # boom (mogelijke dubbelen niet meegeteld) staat 4 m of verder; bomenrijen dus ook
        "solitair": (verder >= 4.0) & (soort != "bos"),
    }, index=bomen.index)
