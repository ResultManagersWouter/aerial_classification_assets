"""Standaardopmaak voor QGIS, in de GeoPackage zelf.

QGIS zoekt bij het openen van een laag in een GeoPackage naar een tabel layer_styles en
neemt de stijl daaruit over als die als standaard is gemarkeerd. Zo opent de uitvoer meteen
gekleurd per status, met twee acties op elk punt: het Amsterdamse panorama en Google Street
View. Kies in QGIS de knop 'Acties uitvoeren' (het tandwiel met de bliksem) en klik op een
punt, of open de link vanuit het identificatievenster.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

ACTIES = (("Panorama Amsterdam", "Panorama", "panorama_beeld"),
          ("Google Street View", "Street View", "google_streetview"),
          ("Google Maps", "Maps", "google_maps"))


def _rgb(kleur: str) -> str:
    kleur = kleur.lstrip("#")
    return ",".join(str(int(kleur[i:i + 2], 16)) for i in (0, 2, 4)) + ",255"


def _qml(veld: str, kleuren: dict[str, str], velden: set[str], lijn: bool = False) -> str:
    categorieen, symbolen = [], []
    for i, (waarde, kleur) in enumerate(kleuren.items()):
        waarde_xml = waarde.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        categorieen.append(f'<category value="{waarde_xml}" symbol="{i}" label="{waarde_xml}" render="true"/>')
        if lijn:
            laag = (f'<layer class="SimpleLine" enabled="1" pass="0" locked="0"><Option type="Map">'
                    f'<Option name="line_color" value="{_rgb(kleur)}" type="QString"/>'
                    f'<Option name="line_width" value="0.8" type="QString"/></Option></layer>')
            symbolen.append(f'<symbol type="line" name="{i}" alpha="1" clip_to_extent="1" force_rhr="0">{laag}</symbol>')
        else:
            laag = (f'<layer class="SimpleMarker" enabled="1" pass="0" locked="0"><Option type="Map">'
                    f'<Option name="color" value="{_rgb(kleur)}" type="QString"/>'
                    f'<Option name="name" value="circle" type="QString"/>'
                    f'<Option name="size" value="2.8" type="QString"/>'
                    f'<Option name="outline_color" value="255,255,255,255" type="QString"/>'
                    f'<Option name="outline_width" value="0.3" type="QString"/></Option></layer>')
            symbolen.append(f'<symbol type="marker" name="{i}" alpha="1" clip_to_extent="1" force_rhr="0">{laag}</symbol>')
    acties = "".join(
        f'<actionsetting type="5" name="{naam}" shortTitle="{kort}" action="[%{kolom}%]" capture="0" '
        f'notificationMessage="" isEnabledOnlyWhenEditable="0" icon="" id="{{{uuid.uuid4()}}}">'
        f'<actionScope id="Feature"/><actionScope id="Canvas"/><actionScope id="Field"/></actionsetting>'
        for naam, kort, kolom in ACTIES if kolom in velden)
    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
        '<qgis version="3.34" styleCategories="AllStyleCategories">'
        f'<renderer-v2 type="categorizedSymbol" attr="{veld}" symbollevels="0" enableorderby="0" forceraster="0">'
        f'<categories>{"".join(categorieen)}</categories><symbols>{"".join(symbolen)}</symbols></renderer-v2>'
        f'<attributeactions><defaultAction key="Canvas" value="{{00000000-0000-0000-0000-000000000000}}"/>{acties}'
        "</attributeactions></qgis>"
    )


def zet_stijl(pad: Path, lagen: list[str], veld: str, kleuren: dict[str, str]) -> None:
    """Zet voor elke laag een standaardstijl in de GeoPackage: categorie per `veld`, met link-acties."""
    db = sqlite3.connect(pad)
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT,
            f_table_catalog TEXT(256), f_table_schema TEXT(256), f_table_name TEXT(256),
            f_geometry_column TEXT(256), styleName TEXT(30), styleQML TEXT, styleSLD TEXT,
            useAsDefault BOOLEAN, description TEXT, owner TEXT(30), ui TEXT(30),
            update_time DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        db.execute("INSERT OR IGNORE INTO gpkg_contents (table_name, data_type, identifier) "
                   "VALUES ('layer_styles', 'attributes', 'layer_styles')")
        for laag in lagen:
            kolom = db.execute("SELECT column_name, geometry_type_name FROM gpkg_geometry_columns WHERE table_name=?",
                               (laag,)).fetchone()
            if kolom is None:
                continue
            velden = {r[1] for r in db.execute(f'PRAGMA table_info("{laag}")')}
            qml = _qml(veld, kleuren, velden, lijn="LINE" in kolom[1].upper())
            db.execute("DELETE FROM layer_styles WHERE f_table_name=?", (laag,))
            db.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column, "
                       "styleName, styleQML, styleSLD, useAsDefault, description, owner, ui) "
                       "VALUES ('', '', ?, ?, ?, ?, '', 1, 'gemaakt door de analyse', '', '')",
                       (laag, kolom[0], laag, qml))
        db.commit()
    finally:
        db.close()
