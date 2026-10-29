"""Straatbeeld bij een punt: het panorama van Amsterdam en Google Street View.

Amsterdam rijdt elk jaar met een panoramawagen door de stad en zet de beelden open online,
met opnamedatum. Per punt zoeken we het nieuwste panorama binnen 30 m, rekenen de
kijkrichting van de wagen naar het punt uit, en bewaren een link naar een uitsnede die
recht op het punt kijkt. Klik je die aan, dan zie je de plek zoals hij op die datum was.

De Google-link opent Street View op dezelfde plek met dezelfde kijkrichting. Google kiest
zelf het dichtstbijzijnde beeld, dus de datum is daar niet vooraf bekend; in Street View
staat hij linksonder.

Het opzoeken kost een verzoek per punt, ongeveer twintig per seconde. De uitkomst wordt per
doel bewaard in de assetmap (straatbeeld/panorama_<doel>.csv), zodat een volgende run
alleen nieuwe punten opzoekt.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from luchtfoto_objecten import opslag

logger = logging.getLogger(__name__)

PANORAMA_API = "https://api.data.amsterdam.nl/panorama"
ZOEKSTRAAL_M = 30
NAAR_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def _sessie() -> requests.Session:
    sessie = requests.Session()
    herhaling = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    sessie.mount("https://", HTTPAdapter(max_retries=herhaling, pool_maxsize=16))
    return sessie


def _kijkrichting(lon_van, lat_van, lon_naar, lat_naar) -> float:
    """Kompasrichting in graden van het panorama naar het punt."""
    dlon = math.radians(lon_naar - lon_van)
    lat1, lat2 = math.radians(lat_van), math.radians(lat_naar)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def google_streetview(lat: float, lon: float, kijkrichting: float | None = None) -> str:
    url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.6f},{lon:.6f}"
    return url + (f"&heading={kijkrichting:.0f}" if kijkrichting is not None and np.isfinite(kijkrichting) else "")


def google_maps(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def links(punten) -> pd.DataFrame:
    """Google-links voor elk punt, zonder iets op te vragen."""
    lon, lat = NAAR_WGS84.transform(punten.geometry.x.values, punten.geometry.y.values)
    return pd.DataFrame({
        "punt_id": punten["punt_id"].values, "lat": np.round(lat, 7), "lon": np.round(lon, 7),
        "google_maps": [google_maps(a, o) for a, o in zip(lat, lon)],
    })


def _zoek_panorama(sessie: requests.Session, punt_id: str, lat: float, lon: float) -> dict:
    try:
        antwoord = sessie.get(f"{PANORAMA_API}/panoramas/", timeout=60, params={
            "near": f"{lon},{lat}", "radius": ZOEKSTRAAL_M, "srid": 4326, "newest_in_range": "true", "page_size": 1})
        antwoord.raise_for_status()
        gevonden = antwoord.json()["_embedded"]["panoramas"]
    except Exception as fout:
        logger.debug("Panorama bij %s niet op te zoeken: %s", punt_id, fout)
        return {"punt_id": punt_id}
    if not gevonden:
        return {"punt_id": punt_id, "panorama_datum": "geen binnen 30 m"}
    pano = gevonden[0]
    plon, plat = pano["geometry"]["coordinates"][:2]
    richting = _kijkrichting(plon, plat, lon, lat)
    return {
        "punt_id": punt_id,
        "panorama_datum": pano["timestamp"][:10],
        "panorama_afstand_m": round(math.dist(*(np.array(NAAR_RD.transform([plon, lon], [plat, lat])).T)), 1),
        "kijkrichting": round(richting),
        "panorama_beeld": f"{PANORAMA_API}/thumbnail/{pano['pano_id']}/?width=1200&heading={richting:.0f}",
        "panorama_360": pano["_links"]["equirectangular_medium"]["href"],
    }


NAAR_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def panorama_bij_punten(punten, doel: str, werkers: int = 8) -> pd.DataFrame:
    """Het nieuwste Amsterdamse panorama bij elk punt, met link en datum; bewaard per doel."""
    basis = links(punten)
    pad = opslag.map_in_assets("straatbeeld") / f"panorama_{doel}.csv"
    bekend = pd.read_csv(pad) if pad.exists() else pd.DataFrame(columns=["punt_id"])
    nieuw = basis[~basis["punt_id"].isin(bekend["punt_id"])]
    if len(nieuw):
        logger.info("Panorama opzoeken voor %s punten", f"{len(nieuw):,}")
        sessie = _sessie()
        with ThreadPoolExecutor(werkers) as pool:
            gevonden = list(tqdm(pool.map(lambda r: _zoek_panorama(sessie, r.punt_id, r.lat, r.lon),
                                          nieuw.itertuples()), total=len(nieuw), unit="punt", desc="panorama"))
        bekend = pd.concat([bekend, pd.DataFrame(gevonden)], ignore_index=True)
        tijdelijk = opslag.LOKAAL_WERK / pad.name
        tijdelijk.parent.mkdir(parents=True, exist_ok=True)
        bekend.to_csv(tijdelijk, index=False)
        opslag.veilig_kopieren(tijdelijk, pad)
        tijdelijk.unlink()
    uit = basis.merge(bekend, on="punt_id", how="left")
    for kolom in ("panorama_datum", "panorama_afstand_m", "kijkrichting", "panorama_beeld", "panorama_360"):
        if kolom not in uit:
            uit[kolom] = np.nan
    uit["google_streetview"] = [google_streetview(a, o, r) for a, o, r in zip(uit["lat"], uit["lon"], uit["kijkrichting"])]
    return uit
