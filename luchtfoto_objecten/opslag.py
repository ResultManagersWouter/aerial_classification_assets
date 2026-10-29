"""Gedeelde opslag voor data die lang duurt om op te halen: "data assets Amsterdam".

Alles wat een analyse uit een open bron haalt en wat minuten tot uren kost, komt hier één
keer terecht en wordt daarna hergebruikt, ook door collega's en door andere projecten. De
map staat in de gedeelde OneDrive van Amsterdam - General:

    bomen/stamgegevens/stamgegevens_<jjjjmmdd>.gpkg      bomen-WFS, alle boompunten
    bomen/kapenherplant/kapenherplant_<jjjjmmdd>.gpkg    bomen-WFS, kap- en herplanttrajecten
    ahn/AHN4/objecthoogte/AHN4_objecthoogte_<x>_<y>.tif  hoogte boven maaiveld, 1 km-tegels
    ahn/AHN5/objecthoogte/AHN5_objecthoogte_<x>_<y>.tif
    ahn/AHN5/kronen/                                     boomtoppen en kroonvlakken uit het AHN
    luchtfoto/<laag>.gpkg                                PDOK-luchtfototegels als GeoPackage-tegelset
    straatbeeld/panorama_<doel>.csv                      dichtstbijzijnde panorama per boompunt
    afgeleid/<analyse>/                                  per punt gemeten kenmerken
    LEESMIJ.md                                           wat staat waar, met bron en datum

Twee keuzes die uitleg vragen.

Het pad bevat "Result!Managers", en GDAL leest een uitroepteken als scheiding binnen een
zip-archief. Daarom loopt alles via een symlink data/assets in dit project. Gewone Python-
bestandsbewerkingen hebben er geen last van, GDAL wel.

De luchtfototegels staan niet als losse bestanden in OneDrive maar als één GeoPackage per
laag. Tienduizenden kleine bestanden synchroniseren slecht, en zo'n tegelset opent QGIS
direct als rasterlaag. Tijdens een run wordt er in een lokale werkkopie geschreven; pas aan
het eind gaat het bestand in één keer naar OneDrive, zodat de synchronisatie nooit een half
geschreven database oppakt.
"""

from __future__ import annotations

import io
import logging
import shutil
import sqlite3
import threading
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import Affine
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[1]
ASSETS_ONEDRIVE = Path(
    "/Users/wks/Library/CloudStorage/OneDrive-SharedLibraries-Result!Managers/"
    "Amsterdam - General/data assets Amsterdam"
)
ASSETS_LINK = PROJECT / "data" / "assets"
LOKAAL_WERK = PROJECT / "data" / "cache" / "werkkopie"


def assets() -> Path:
    """De assetmap via de symlink, met een lokale terugval als OneDrive er niet is."""
    if ASSETS_LINK.exists():
        return ASSETS_LINK
    if ASSETS_ONEDRIVE.exists():
        ASSETS_LINK.parent.mkdir(parents=True, exist_ok=True)
        ASSETS_LINK.symlink_to(ASSETS_ONEDRIVE, target_is_directory=True)
        return ASSETS_LINK
    lokaal = PROJECT / "data" / "assets_lokaal"
    logger.warning("De OneDrive-map %s is niet gevonden; we bewaren lokaal in %s", ASSETS_ONEDRIVE, lokaal)
    lokaal.mkdir(parents=True, exist_ok=True)
    return lokaal


def map_in_assets(*delen: str) -> Path:
    pad = assets().joinpath(*delen)
    pad.mkdir(parents=True, exist_ok=True)
    return pad


def veilig_kopieren(bron: Path, doel: Path) -> None:
    """Eerst naar een tijdelijke naam en dan hernoemen, zodat OneDrive nooit een half bestand ziet."""
    doel.parent.mkdir(parents=True, exist_ok=True)
    tijdelijk = doel.with_name(doel.name + ".deel")
    shutil.copyfile(bron, tijdelijk)
    tijdelijk.replace(doel)


# ------------------------------------------------------------------- WFS-exports


def wfs_export(naam: str, ophalen, max_leeftijd_dagen: int = 7):
    """De nieuwste export van een WFS-laag uit de opslag, of een verse als die te oud is.

    `ophalen` is een functie zonder argumenten die een GeoDataFrame teruggeeft. Een verse
    export wordt met de datum in de naam bewaard, oudere blijven staan voor vergelijking.
    """
    import geopandas as gpd

    map_ = map_in_assets("bomen", naam)
    bestaand = sorted(map_.glob(f"{naam}_*.gpkg"))
    if bestaand:
        nieuwste = bestaand[-1]
        datum = datetime.strptime(nieuwste.stem.split("_")[-1], "%Y%m%d").date()
        if (date.today() - datum).days <= max_leeftijd_dagen:
            logger.info("%s uit de opslag: %s", naam, nieuwste.name)
            return gpd.read_file(nieuwste)
    gdf = ophalen()
    lokaal = LOKAAL_WERK / f"{naam}_{date.today():%Y%m%d}.gpkg"
    lokaal.parent.mkdir(parents=True, exist_ok=True)
    from luchtfoto_objecten.geo_hulp import maak_schrijfbaar

    maak_schrijfbaar(gdf).to_file(lokaal, layer=naam, driver="GPKG")
    veilig_kopieren(lokaal, map_ / lokaal.name)
    lokaal.unlink()
    logger.info("%s opgehaald en bewaard: %s features", naam, f"{len(gdf):,}")
    return gdf


# ---------------------------------------------------------------------------- AHN

AHN_TEGELS = "https://fsn1.your-objectstorage.com/hwh-ahn"
AHN_PIXEL_M = 0.5
TEGEL_M = 1000


def _ahn_pad(versie: int, tx: int, ty: int) -> Path:
    return map_in_assets("ahn", f"AHN{versie}", "objecthoogte") / f"AHN{versie}_objecthoogte_{tx:06d}_{ty:06d}.tif"


def _ontbrekend_pad(versie: int) -> Path:
    return map_in_assets("ahn", f"AHN{versie}") / "ontbrekende_tegels.txt"


_ontbreekt_slot = threading.Lock()


def _ontbrekende_tegels(versie: int) -> set[str]:
    pad = _ontbrekend_pad(versie)
    return set(pad.read_text().split()) if pad.exists() else set()


def _haal_ahn_bestand(url: str, pogingen: int = 4) -> bytes | None:
    for poging in range(1, pogingen + 1):
        try:
            antwoord = requests.get(url, timeout=600)
            if antwoord.status_code in (403, 404):
                return None  # de objectopslag geeft 403 voor een tegel die niet bestaat
            antwoord.raise_for_status()
            return antwoord.content
        except requests.RequestException as fout:
            if poging == pogingen:
                raise
            logger.warning("AHN-download haperde (%s), opnieuw over %ss", fout, 10 * poging)
            time.sleep(10 * poging)
    return None


def objecthoogte_tegel(versie: int, tx: int, ty: int) -> Path | None:
    """Objecthoogte (DSM min DTM) van één kilometertegel, uit de opslag of vers berekend.

    Bewaard als uint16 in decimeters met deflate: 10 cm is ruim nauwkeurig genoeg voor
    bomen, en het scheelt de helft in omvang. Gaten in het maaiveldmodel (panden, water)
    worden vanaf de rand gevuld, zoals in hoogte.py; waar de bovenkant niets heeft wordt 0.
    """
    pad = _ahn_pad(versie, tx, ty)
    if pad.exists():
        return pad
    sleutel = f"{tx}_{ty}"
    if sleutel in _ontbrekende_tegels(versie):
        return None
    basis = f"{AHN_TEGELS}/AHN{versie}_KM"
    dsm = _haal_ahn_bestand(f"{basis}/03a_DSM_50cm/AHN{versie}_R_{tx:06d}_{ty:06d}.TIF")
    dtm = _haal_ahn_bestand(f"{basis}/02a_DTM_50cm/AHN{versie}_M_{tx:06d}_{ty:06d}.TIF") if dsm else None
    if dsm is None or dtm is None:
        with _ontbreekt_slot:
            with _ontbrekend_pad(versie).open("a") as bestand:
                bestand.write(sleutel + "\n")
        return None

    from luchtfoto_objecten.hoogte import _vul_gaten

    vlakken = []
    for inhoud in (dsm, dtm):
        with rasterio.open(io.BytesIO(inhoud)) as bron:
            vlak = bron.read(1).astype(np.float32)
            if bron.nodata is not None:
                vlak[vlak == bron.nodata] = np.nan
            transform = bron.transform
        vlakken.append(vlak)
    bovenkant, maaiveld = vlakken
    if not np.isfinite(maaiveld).any():
        hoogte = np.zeros_like(bovenkant)
    else:
        hoogte = np.nan_to_num(bovenkant - _vul_gaten(maaiveld), nan=0.0)
    dm = np.clip(np.round(hoogte * 10), 0, 65000).astype(np.uint16)

    tijdelijk = LOKAAL_WERK / pad.name
    tijdelijk.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tijdelijk, "w", driver="GTiff", width=dm.shape[1], height=dm.shape[0], count=1,
                       dtype="uint16", crs="EPSG:28992", transform=transform, compress="deflate", predictor=2,
                       tiled=True, blockxsize=256, blockysize=256) as uit:
        uit.write(dm, 1)
        uit.update_tags(eenheid="decimeter boven maaiveld", bron=f"AHN{versie} via {AHN_TEGELS}",
                        gemaakt=f"{date.today():%Y-%m-%d}")
    veilig_kopieren(tijdelijk, pad)
    tijdelijk.unlink()
    return pad


def zorg_voor_ahn(versie: int, tegels: set[tuple[int, int]], werkers: int = 8) -> None:
    """Haalt de ontbrekende kilometertegels parallel op; wat er al is kost niets."""
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    ontbreekt = _ontbrekende_tegels(versie)
    nodig = sorted(t for t in tegels if not _ahn_pad(versie, *t).exists() and f"{t[0]}_{t[1]}" not in ontbreekt)
    if not nodig:
        return
    logger.info("AHN%s: %s kilometertegels ophalen (de eerste keer duurt dit lang)", versie, len(nodig))
    with ThreadPoolExecutor(werkers) as pool:
        list(tqdm(pool.map(lambda t: objecthoogte_tegel(versie, *t), nodig), total=len(nodig),
                  unit="tegel", desc=f"AHN{versie}"))


def kilometertegels(xs, ys, marge: float = 0.0) -> set[tuple[int, int]]:
    """De kilometertegels (linksonder) waar punten in vallen, eventueel met een rand erbij."""
    tegels = set()
    for dx in ((-marge, 0, marge) if marge else (0,)):
        for dy in ((-marge, 0, marge) if marge else (0,)):
            tegels |= set(zip(((np.asarray(xs) + dx) // TEGEL_M * TEGEL_M).astype(int),
                              ((np.asarray(ys) + dy) // TEGEL_M * TEGEL_M).astype(int)))
    return tegels


def objecthoogte_venster(versie: int, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, Affine] | None:
    """Objecthoogte in meters voor een willekeurige bbox, samengesteld uit de opgeslagen tegels.

    Haalt ontbrekende tegels eerst op. Geeft None als er voor het hele venster geen AHN is.
    """
    xmin, ymin, xmax, ymax = (round(w / AHN_PIXEL_M) * AHN_PIXEL_M for w in bbox)
    breedte, hoogte = round((xmax - xmin) / AHN_PIXEL_M), round((ymax - ymin) / AHN_PIXEL_M)
    uit = np.full((hoogte, breedte), np.nan, dtype=np.float32)
    gevonden = False
    for tx in range(int(xmin // TEGEL_M * TEGEL_M), int(xmax), TEGEL_M):
        for ty in range(int(ymin // TEGEL_M * TEGEL_M), int(ymax), TEGEL_M):
            pad = objecthoogte_tegel(versie, tx, ty)
            if pad is None:
                continue
            x0, y0, x1, y1 = max(xmin, tx), max(ymin, ty), min(xmax, tx + TEGEL_M), min(ymax, ty + TEGEL_M)
            with rasterio.open(pad) as bron:
                venster = from_bounds(x0, y0, x1, y1, bron.transform).round_offsets().round_lengths()
                stuk = bron.read(1, window=venster).astype(np.float32) / 10.0
            rij0, kol0 = round((ymax - y1) / AHN_PIXEL_M), round((x0 - xmin) / AHN_PIXEL_M)
            uit[rij0:rij0 + stuk.shape[0], kol0:kol0 + stuk.shape[1]] = stuk[:hoogte - rij0, :breedte - kol0]
            gevonden = True
    if not gevonden:
        return None
    return uit, Affine(AHN_PIXEL_M, 0, xmin, 0, -AHN_PIXEL_M, ymax)


# --------------------------------------------------------- luchtfoto als tegelset

# De RD-tegelmatrix van PDOK: linksboven en de schaal op niveau 0, zie wmts.py
_RD_LINKSBOVEN = (-285401.92, 903401.92)
_SPAN_NIVEAU_NUL = 880803.84
_MAX_ZOOM = 19


class Tegelopslag:
    """Luchtfototegels van één WMTS-laag als GeoPackage-tegelset, te openen in QGIS.

    De tegels houden de nummering van de PDOK-tegelmatrix in RD, dus een tegel die uit de
    WMTS komt gaat er ongewijzigd in. Er wordt geschreven in een lokale werkkopie; met
    bewaar() gaat het bestand naar de assetmap.
    """

    def __init__(self, laag: str) -> None:
        self.laag = laag
        self.tabel = "tegels"
        self.doel = map_in_assets("luchtfoto") / f"{laag}.gpkg"
        self.lokaal = LOKAAL_WERK / f"{laag}.gpkg"
        self.lokaal.parent.mkdir(parents=True, exist_ok=True)
        if not self.lokaal.exists() and self.doel.exists():
            logger.info("Tegelset %s uit de opslag halen", laag)
            shutil.copyfile(self.doel, self.lokaal)
        nieuw = not self.lokaal.exists()
        self._slot = threading.Lock()
        self._db = sqlite3.connect(self.lokaal, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        if nieuw:
            self._maak_aan()
        self._ongeschreven = 0
        self.gewijzigd = False

    def _maak_aan(self) -> None:
        from pyproj import CRS

        wkt = CRS.from_epsg(28992).to_wkt("WKT1_GDAL")
        db = self._db
        db.execute("PRAGMA application_id = 1196444487")   # 'GPKG'
        db.execute("PRAGMA user_version = 10300")
        db.executescript(f"""
            CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
                organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL, description TEXT);
            CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
                identifier TEXT UNIQUE, description TEXT DEFAULT '',
                last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER);
            CREATE TABLE gpkg_tile_matrix_set (table_name TEXT NOT NULL PRIMARY KEY, srs_id INTEGER NOT NULL,
                min_x DOUBLE NOT NULL, min_y DOUBLE NOT NULL, max_x DOUBLE NOT NULL, max_y DOUBLE NOT NULL);
            CREATE TABLE gpkg_tile_matrix (table_name TEXT NOT NULL, zoom_level INTEGER NOT NULL,
                matrix_width INTEGER NOT NULL, matrix_height INTEGER NOT NULL, tile_width INTEGER NOT NULL,
                tile_height INTEGER NOT NULL, pixel_x_size DOUBLE NOT NULL, pixel_y_size DOUBLE NOT NULL,
                CONSTRAINT pk_ttm PRIMARY KEY (table_name, zoom_level));
            CREATE TABLE {self.tabel} (id INTEGER PRIMARY KEY AUTOINCREMENT, zoom_level INTEGER NOT NULL,
                tile_column INTEGER NOT NULL, tile_row INTEGER NOT NULL, tile_data BLOB NOT NULL,
                UNIQUE (zoom_level, tile_column, tile_row));
        """)
        db.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", [
            ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None),
            ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None),
            ("Amersfoort / RD New", 28992, "EPSG", 28992, wkt, None),
        ])
        x0, y1 = _RD_LINKSBOVEN
        x1, y0 = x0 + _SPAN_NIVEAU_NUL, y1 - _SPAN_NIVEAU_NUL
        db.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, description, min_x, min_y, max_x, "
                   "max_y, srs_id) VALUES (?, 'tiles', ?, ?, ?, ?, ?, ?, 28992)",
                   (self.tabel, f"luchtfoto {self.laag}", f"PDOK WMTS-laag {self.laag}", x0, y0, x1, y1))
        db.execute("INSERT INTO gpkg_tile_matrix_set VALUES (?, 28992, ?, ?, ?, ?)", (self.tabel, x0, y0, x1, y1))
        for zoom in range(_MAX_ZOOM + 1):
            res = _SPAN_NIVEAU_NUL / 256 / 2 ** zoom
            db.execute("INSERT INTO gpkg_tile_matrix VALUES (?,?,?,?,256,256,?,?)",
                       (self.tabel, zoom, 2 ** zoom, 2 ** zoom, res, res))
        db.commit()

    def lees(self, zoom: int, kolom: int, rij: int) -> bytes | None:
        with self._slot:
            rij_ = self._db.execute(f"SELECT tile_data FROM {self.tabel} WHERE zoom_level=? AND tile_column=? "
                                    "AND tile_row=?", (zoom, kolom, rij)).fetchone()
        return rij_[0] if rij_ else None

    def aanwezig(self, zoom: int) -> set[tuple[int, int]]:
        with self._slot:
            return set(self._db.execute(f"SELECT tile_column, tile_row FROM {self.tabel} WHERE zoom_level=?",
                                        (zoom,)).fetchall())

    def schrijf(self, zoom: int, kolom: int, rij: int, inhoud: bytes) -> None:
        with self._slot:
            self._db.execute(f"INSERT OR REPLACE INTO {self.tabel} (zoom_level, tile_column, tile_row, tile_data) "
                             "VALUES (?,?,?,?)", (zoom, kolom, rij, inhoud))
            self._ongeschreven += 1
            self.gewijzigd = True
            if self._ongeschreven >= 500:
                self._db.commit()
                self._ongeschreven = 0

    def importeer_map(self, map_: Path, zoom: int) -> int:
        """Neemt losse tegels uit de oude bestandscache (data/cache/wmts/<laag>/<zoom>) over."""
        bron = map_ / f"{zoom:02d}"
        if not bron.exists():
            return 0
        al = self.aanwezig(zoom)
        aantal = 0
        nu = time.time()
        for pad in bron.glob("*/*.jpeg"):
            kolom, rij = int(pad.parent.name), int(pad.stem)
            # Een tegel die net geschreven wordt (door een andere run) nog even laten liggen
            if (kolom, rij) not in al and nu - pad.stat().st_mtime > 120:
                self.schrijf(zoom, kolom, rij, pad.read_bytes())
                aantal += 1
        self._db.commit()
        return aantal

    def bewaar(self) -> None:
        """Schrijft af en zet de tegelset in de assetmap, alleen als er iets nieuws in staat."""
        with self._slot:
            self._db.commit()
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if self.gewijzigd or not self.doel.exists():
            veilig_kopieren(self.lokaal, self.doel)
            logger.info("Tegelset %s bewaard in de opslag (%.0f MB)", self.laag, self.doel.stat().st_size / 1e6)
            self.gewijzigd = False


# ------------------------------------------------------------------------ LEESMIJ

OMSCHRIJVING = {
    "bomen/stamgegevens": ("Alle boompunten uit de bomen-WFS (laag stamgegevens), openbare velden",
                           "https://api.data.amsterdam.nl/v1/wfs/bomen"),
    "bomen/kapenherplant": ("Kap- en herplanttrajecten uit de bomen-WFS: kapdatum, herplantinspectie, planten",
                            "https://api.data.amsterdam.nl/v1/wfs/bomen"),
    "ahn/AHN4/objecthoogte": ("Hoogte boven maaiveld in decimeters (uint16), 0,5 m, AHN4 ingewonnen 2020-2022",
                              AHN_TEGELS),
    "ahn/AHN5/objecthoogte": ("Hoogte boven maaiveld in decimeters (uint16), 0,5 m, AHN5, Amsterdam rond voorjaar 2023",
                              AHN_TEGELS),
    "ahn/AHN5/kronen": ("Boomtoppen en kroonvlakken, afgeleid uit de AHN5-objecthoogte (lokale maxima en watershed)",
                        "afgeleid"),
    "ahn/AHN4/kronen": ("Boomtoppen en kroonvlakken, afgeleid uit de AHN4-objecthoogte", "afgeleid"),
    "luchtfoto": ("PDOK-luchtfototegels per laag als GeoPackage-tegelset in RD, direct te openen in QGIS",
                  "https://service.pdok.nl/hwh/luchtfotocir/wmts/v1_0"),
    "straatbeeld": ("Per boompunt het nieuwste panorama van Amsterdam, met opnamedatum, kijkrichting en link",
                    "https://api.data.amsterdam.nl/panorama"),
    "afgeleid": ("Per punt gemeten kenmerken (hoogte, kroon, vegetatiegetal), om analyses niet opnieuw te meten",
                 "afgeleid"),
}


def werk_leesmij_bij() -> Path:
    """Schrijft LEESMIJ.md met per map wat erin staat, de bron, de omvang en de datum."""
    basis = assets()
    regels = [
        "# Data assets Amsterdam",
        "",
        "Open data die lang duurt om op te halen, één keer opgehaald en hier bewaard voor hergebruik. "
        "Bijgehouden door `luchtfoto_objecten/opslag.py` in het project aerial_classification_assets; "
        "deze lijst wordt bij elke run opnieuw geschreven.",
        "",
        "Let op bij gebruik in eigen code: het pad bevat een uitroepteken (Result!Managers) en GDAL leest "
        "dat als zip-scheiding. Werk via een symlink zonder uitroepteken, of kopieer het bestand eerst.",
        "",
        "| map | inhoud | bron | bestanden | omvang | laatst bijgewerkt |",
        "|---|---|---|---|---|---|",
    ]
    for sleutel, (inhoud, bron) in OMSCHRIJVING.items():
        map_ = basis / sleutel
        if not map_.exists():
            continue
        bestanden = [p for p in map_.rglob("*") if p.is_file()]
        if not bestanden:
            continue
        omvang = sum(p.stat().st_size for p in bestanden) / 1e6
        laatst = datetime.fromtimestamp(max(p.stat().st_mtime for p in bestanden))
        regels.append(f"| `{sleutel}` | {inhoud} | {bron} | {len(bestanden)} | {omvang:,.0f} MB | {laatst:%Y-%m-%d} |")
    pad = basis / "LEESMIJ.md"
    pad.write_text("\n".join(regels) + "\n", encoding="utf-8")
    return pad
