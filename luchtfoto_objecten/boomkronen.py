"""Boomkronen uit het AHN: toppen zoeken en kronen afbakenen.

Een boom is in een hoogtemodel een heuvel: een top met daaromheen een kroon die naar de
rand toe zakt. De objecthoogte wordt licht gladgestreken (een halve meter), de toppen zijn
de lokale maxima boven 3 m met minstens 1,5 m tussen twee toppen, en een watershed vanuit
die toppen verdeelt alles boven 2 m over de kronen. Dat is de gangbare aanpak voor
individuele boomdetectie uit lidar.

Gemeten op een kilometer in Zuid (AHN5, 1.224 geregistreerde bomen): van de losstaande
straatbomen van voor 2020 staat 96% in een gevonden kroon en heeft 83% een top binnen 2 m
van het geregistreerde punt; 3% deelt zijn kroon met een ander geregistreerd punt. In
bomenrijen zijn dat 66% en 23%, want daar groeien kronen in elkaar.

Elke kroon krijgt een vast id: de positie van zijn top in het landelijke 0,5m-raster. Een
kroon die over een tegelgrens hangt heeft zo in beide tegels hetzelfde id. Per tegel wordt
gerekend met 20 m rand van de buren, zodat kronen aan de rand niet afgekapt worden.

Bewaard in de assetmap onder ahn/AHN<versie>/kronen/: per tegel een raster met het
kroonnummer per pixel, en één GeoPackage met alle toppen en hun kroonmaten.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from scipy import ndimage as ndi
from tqdm import tqdm

from luchtfoto_objecten import opslag

logger = logging.getLogger(__name__)

GLAD_SIGMA_PX = 1.0       # 0,5 m, tegen ruis van losse takken
TOP_MIN_HOOGTE_M = 3.0
TOP_MIN_AFSTAND_PX = 3    # 1,5 m tussen twee toppen
KROON_MIN_HOOGTE_M = 2.0
RAND_M = 20.0
ZOEKSTRAAL_M = 1.5        # ligt een punt net buiten een kroon, dan telt de kroon binnen deze afstand


def _kroonmap(versie: int):
    return opslag.map_in_assets("ahn", f"AHN{versie}", "kronen")


def _labelpad(versie: int, tx: int, ty: int):
    return opslag.map_in_assets("ahn", f"AHN{versie}", "kronen", "tegels") / f"kronen_{tx:06d}_{ty:06d}.tif"


def _globaal_id(rij: np.ndarray, kolom: np.ndarray, transform: Affine) -> np.ndarray:
    """Vast id van een top: rij en kolom in een landelijk 0,5m-raster vanaf RD (0, 650000)."""
    x = transform.c + (kolom + 0.5) * transform.a
    y = transform.f + (rij + 0.5) * transform.e
    return (np.round((650000 - y) / 0.5).astype(np.int64) * 1_000_000 + np.round(x / 0.5).astype(np.int64))


def kronen_tegel(versie: int, tx: int, ty: int) -> pd.DataFrame | None:
    """Toppen en kronen voor één kilometertegel; schrijft het kroonraster en geeft de toppen terug."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    gelezen = opslag.objecthoogte_venster(versie, (tx - RAND_M, ty - RAND_M, tx + 1000 + RAND_M, ty + 1000 + RAND_M))
    if gelezen is None:
        return None
    hoogte, transform = gelezen
    hoogte = np.nan_to_num(hoogte, nan=0.0)
    glad = ndi.gaussian_filter(hoogte, GLAD_SIGMA_PX)
    toppen = peak_local_max(glad, min_distance=TOP_MIN_AFSTAND_PX, threshold_abs=TOP_MIN_HOOGTE_M,
                            exclude_border=False)
    if len(toppen) == 0:
        return pd.DataFrame()
    markers = np.zeros(glad.shape, dtype=np.int32)
    markers[toppen[:, 0], toppen[:, 1]] = np.arange(1, len(toppen) + 1)
    labels = watershed(-glad, markers, mask=glad > KROON_MIN_HOOGTE_M)

    index = np.arange(1, len(toppen) + 1)
    oppervlak = ndi.sum(np.ones_like(hoogte), labels, index) * transform.a ** 2
    volume = ndi.sum(hoogte, labels, index) * transform.a ** 2
    top_hoogte = hoogte[toppen[:, 0], toppen[:, 1]]
    ids = _globaal_id(toppen[:, 0], toppen[:, 1], transform)

    # Alleen de kern van de tegel bewaren; de rand hoort bij de buurtegel
    rand = round(RAND_M / transform.a)
    kern = labels[rand:rand + 2000, rand:rand + 2000]
    opzoek = np.concatenate([[0], np.arange(1, len(toppen) + 1)])
    kern_lokaal = opzoek[kern].astype(np.uint32)
    kern_transform = Affine(transform.a, 0, tx, 0, transform.e, ty + 1000)
    pad = _labelpad(versie, tx, ty)
    tijdelijk = opslag.LOKAAL_WERK / f"AHN{versie}_{pad.name}"
    tijdelijk.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tijdelijk, "w", driver="GTiff", width=kern.shape[1], height=kern.shape[0], count=1,
                       dtype="uint32", crs="EPSG:28992", transform=kern_transform, compress="deflate",
                       tiled=True, nodata=0) as uit:
        uit.write(kern_lokaal, 1)
    opslag.veilig_kopieren(tijdelijk, pad)
    tijdelijk.unlink()

    x = transform.c + (toppen[:, 1] + 0.5) * transform.a
    y = transform.f + (toppen[:, 0] + 0.5) * transform.e
    toppen_tabel = pd.DataFrame({
        "kroon_id": ids, "lokaal": index, "tegel": f"{tx}_{ty}", "x": x, "y": y,
        "hoogte_m": np.round(top_hoogte, 1), "oppervlak_m2": np.round(oppervlak, 2), "volume_m3": np.round(volume, 1),
    })
    # Alle toppen die in de kern een kroonstuk hebben, ook als de top zelf in de rand ligt
    in_kern = np.isin(index, np.unique(kern[kern > 0]))
    return toppen_tabel[in_kern]


def zorg_voor_kronen(versie: int, tegels: set[tuple[int, int]], werkers: int = 4) -> pd.DataFrame:
    """Kronen voor alle tegels, uit de opslag waar ze al zijn; geeft de toppentabel terug."""
    import geopandas as gpd

    opslag.zorg_voor_ahn(versie, {(tx + dx, ty + dy) for tx, ty in tegels for dx in (-1000, 0, 1000)
                                  for dy in (-1000, 0, 1000)})
    toppen_pad = _kroonmap(versie) / f"boomtoppen_AHN{versie}.gpkg"
    bekend = pd.DataFrame()
    if toppen_pad.exists():
        bekend = pd.DataFrame(gpd.read_file(toppen_pad, ignore_geometry=True))
    gedaan = set(bekend["tegel"]) if len(bekend) else set()
    nodig = sorted(t for t in tegels if f"{t[0]}_{t[1]}" not in gedaan or not _labelpad(versie, *t).exists())
    if nodig:
        logger.info("AHN%s: kronen afbakenen voor %s tegels", versie, len(nodig))
        nieuw = []
        with ThreadPoolExecutor(werkers) as pool:
            for tabel in tqdm(pool.map(lambda t: kronen_tegel(versie, *t), nodig), total=len(nodig), unit="tegel",
                              desc=f"kronen AHN{versie}"):
                if tabel is not None and len(tabel):
                    nieuw.append(tabel)
        klaar = {f"{t[0]}_{t[1]}" for t in nodig}
        bekend = pd.concat([bekend[~bekend["tegel"].isin(klaar)] if len(bekend) else bekend] + nieuw,
                           ignore_index=True)
        gdf = gpd.GeoDataFrame(bekend, geometry=gpd.points_from_xy(bekend["x"], bekend["y"]), crs="EPSG:28992")
        tijdelijk = opslag.LOKAAL_WERK / toppen_pad.name
        if tijdelijk.exists():
            tijdelijk.unlink()
        gdf.to_file(tijdelijk, layer="boomtoppen", driver="GPKG")
        opslag.veilig_kopieren(tijdelijk, toppen_pad)
        tijdelijk.unlink()
    return bekend[bekend["tegel"].isin({f"{t[0]}_{t[1]}" for t in tegels})]


def kronen_bij_punten(versie: int, punten, toppen: pd.DataFrame) -> pd.DataFrame:
    """Per punt de kroon waarin het staat, of de dichtstbijzijnde binnen 1,5 m.

    Geeft per punt: kroon_id, hoogte, oppervlak en volume van die kroon, de afstand tot de
    top, de objecthoogte op het punt zelf, en de afstand tot de dichtstbijzijnde top.
    """
    from scipy.spatial import cKDTree

    p = f"ahn{versie}_"
    tabel = pd.DataFrame({"punt_id": punten["punt_id"].values, "x": punten.geometry.x.values,
                          "y": punten.geometry.y.values})
    tabel["tegel"] = ((tabel["x"] // 1000 * 1000).astype(int).astype(str) + "_" +
                      (tabel["y"] // 1000 * 1000).astype(int).astype(str))
    uit = []
    zoek = round(ZOEKSTRAAL_M / 0.5)
    for tegel, groep in tabel.groupby("tegel"):
        tx, ty = map(int, tegel.split("_"))
        pad = _labelpad(versie, tx, ty)
        if not pad.exists():
            continue
        with rasterio.open(pad) as bron:
            labels = bron.read(1)
            transform = bron.transform
        lokaal_naar_id = toppen[toppen["tegel"] == tegel].set_index("lokaal")["kroon_id"]
        kolom = ((groep["x"].values - transform.c) / transform.a).astype(int)
        rij = ((transform.f - groep["y"].values) / -transform.e).astype(int)
        for punt_id, k, r in zip(groep["punt_id"], kolom, rij):
            venster = labels[max(r - zoek, 0):r + zoek + 1, max(k - zoek, 0):k + zoek + 1]
            label = labels[r, k] if 0 <= r < labels.shape[0] and 0 <= k < labels.shape[1] else 0
            if label == 0 and (venster > 0).any():
                rr, kk = np.nonzero(venster)
                dichtst = np.argmin((rr - min(r, zoek)) ** 2 + (kk - min(k, zoek)) ** 2)
                label = venster[rr[dichtst], kk[dichtst]]
            uit.append((punt_id, lokaal_naar_id.get(int(label), np.nan) if label else np.nan))
    koppeling = pd.DataFrame(uit, columns=["punt_id", "kroon_id"])
    tabel = tabel.merge(koppeling, on="punt_id", how="left")

    maten = toppen.drop_duplicates("kroon_id").set_index("kroon_id")
    tabel[p + "kroon_id"] = tabel["kroon_id"]
    for bron_kolom, doel in (("hoogte_m", "kroon_hoogte_m"), ("oppervlak_m2", "kroon_oppervlak_m2"),
                             ("volume_m3", "kroon_volume_m3")):
        tabel[p + doel] = tabel["kroon_id"].map(maten[bron_kolom])
    tabel[p + "afstand_top_m"] = np.round(np.hypot(tabel["x"] - tabel["kroon_id"].map(maten["x"]),
                                                   tabel["y"] - tabel["kroon_id"].map(maten["y"])), 2)
    alle_toppen = cKDTree(toppen[["x", "y"]].values) if len(toppen) else None
    if alle_toppen is not None:
        afstand, dichtst = alle_toppen.query(tabel[["x", "y"]].values)
        tabel[p + "dichtste_top_m"] = np.round(afstand, 2)
        tabel[p + "dichtste_top_id"] = toppen["kroon_id"].values[dichtst]
    return tabel.drop(columns=["x", "y", "tegel", "kroon_id"])
