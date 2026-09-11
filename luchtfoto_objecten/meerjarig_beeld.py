"""Meerdere infraroodjaargangen samen, voor een steviger oordeel dan één opname geeft.

Eén opname is een momentopname, met alles wat daar toevallig op staat. Een geparkeerde
auto, een natte asfaltplek, de slagschaduw van een gevel, een bouwkeet: stuk voor stuk
dingen die het vegetatiegetal van dat ene jaar vertekenen. Over meerdere jaargangen vallen
ze weg, want ze staan er het jaar erop niet meer.

Wat overblijft is wat er echt staat. Een pixel die in drie van de vier jaargangen begroeid
is, is begroeid; eentje die dat in één jaar was, is ruis of een verandering. Dat verschil
is meteen bruikbaar: elk vlak krijgt mee in hoeveel jaargangen het groen was, en dat is de
zekerheid van die uitspraak.

Elke jaargang heeft zijn eigen kleurzweem, dus de getallen zijn niet zomaar vergelijkbaar.
Ze worden daarom eerst op elkaar gelegd: per jaargang wordt de mediaan van het gebied
verschoven naar die van de nieuwste opname. Dat is een robuuste correctie, gevoelig voor de
zweem en niet voor wat er op die ene foto staat, en daarna betekent dezelfde drempel in elk
jaar hetzelfde.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.wmts import LuchtfotoWMTS, haal_beschikbare_lagen

logger = logging.getLogger(__name__)

IR_PATROON = re.compile(r"^(?:(\d{4})|Actueel)_(?:quick)?ortho(HR|25)IR$")


def infraroodjaargangen(cache_map=None) -> list[str]:
    """Infraroodlagen van nieuw naar oud, per jaar de fijnste.

    'Actueel' telt als het nieuwste jaar. Van elk jaar houden we één laag over, anders
    krijg je hetzelfde beeld twee keer op verschillende resolutie.
    """
    beste: dict[int, tuple[int, str]] = {}
    for naam in haal_beschikbare_lagen(cache_map):
        treffer = IR_PATROON.match(naam)
        if not treffer:
            continue
        jaar = 9999 if treffer.group(1) is None else int(treffer.group(1))
        fijnheid = 0 if treffer.group(2) == "HR" else 1
        if jaar not in beste or fijnheid < beste[jaar][0]:
            beste[jaar] = (fijnheid, naam)
    return [naam for _, (_, naam) in sorted(beste.items(), reverse=True)]


def _uitsnede(laag: str, gebied, instellingen: Instellingen):
    from luchtfoto_objecten.bladstand import _passende_zoom

    wmts = LuchtfotoWMTS(
        laag=laag, zoom=_passende_zoom(laag, instellingen.luchtfoto.zoom),
        cache_map=instellingen.cache_map, max_werkers=instellingen.luchtfoto.max_werkers,
    )
    return wmts.haal_uitsnede(gebied.bbox, toon_voortgang=False)


def _gelijk_raster(bron: np.ndarray, brontransform, doelvorm, doeltransform) -> np.ndarray:
    """Een jaargang op 25cm naar het raster van de nieuwste opname op 8cm."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if bron.shape == doelvorm and brontransform == doeltransform:
        return bron
    doel = np.zeros(doelvorm, dtype=np.float32)
    reproject(
        source=bron, destination=doel,
        src_transform=brontransform, src_crs="EPSG:28992",
        dst_transform=doeltransform, dst_crs="EPSG:28992",
        resampling=Resampling.bilinear,
    )
    return doel


def schat_verschuiving(referentie: np.ndarray, doel: np.ndarray, stap: int = 4) -> tuple[float, float]:
    """Hoeveel ligt deze jaargang verschoven ten opzichte van de nieuwste?

    Jaargangen liggen niet exact op elkaar: de orthorectificatie verschilt per vlucht, en
    hoge objecten helllen bovendien per opname een andere kant op. Zonder correctie zou de
    rand van een haag of kroon per jaar een halve meter opschuiven, en dan telt de consensus
    juist de randen weg in plaats van de ruis.

    De verschuiving wordt met faseconcorrelatie geschat, op een uitgedund raster want dat is
    ruim nauwkeurig genoeg en veel sneller.
    """
    from skimage.registration import phase_cross_correlation

    klein_ref = np.nan_to_num(referentie[::stap, ::stap])
    klein_doel = np.nan_to_num(doel[::stap, ::stap])
    if klein_ref.shape != klein_doel.shape or min(klein_ref.shape) < 16:
        return 0.0, 0.0
    try:
        verschuiving, _, _ = phase_cross_correlation(klein_ref, klein_doel, upsample_factor=4)
    except Exception as fout:
        logger.warning("Verschuiving niet te schatten (%s), we gaan uit van nul", fout)
        return 0.0, 0.0
    return float(verschuiving[0] * stap), float(verschuiving[1] * stap)


def _verschuif(vlak: np.ndarray, verschuiving: tuple[float, float]) -> np.ndarray:
    from scipy.ndimage import shift

    if abs(verschuiving[0]) < 0.01 and abs(verschuiving[1]) < 0.01:
        return vlak
    return shift(vlak, verschuiving, order=1, mode="nearest")


def _vingerafdruk(vlak: np.ndarray) -> float:
    """Goedkope handtekening van een opname, om dezelfde foto onder twee namen te herkennen."""
    monster = vlak[::37, ::41]
    return float(np.nansum(monster.astype(np.float64)))


def consensus(
    gebied, instellingen: Instellingen, doelvorm, doeltransform, drempel: float,
    aantal_jaren: int = 3, binnen: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    """Hoe vaak is elke pixel begroeid, over de laatste jaargangen.

    Geeft het aantal jaren dat een pixel boven de drempel uitkwam, de zekerheid als
    aandeel daarvan, en per jaargang een regeltje met wat er gemeten is. Geeft None terug
    als er niet meer dan één bruikbare jaargang is.
    """
    from luchtfoto_objecten.infrarood import ndvi

    lagen = infraroodjaargangen(instellingen.cache_map)[: aantal_jaren + 2]
    if len(lagen) < 2:
        logger.info("Minder dan twee infraroodjaargangen beschikbaar, we blijven bij één opname")
        return None

    referentie_mediaan = None
    referentiebeeld = None
    vingerafdrukken: list[float] = []
    pixelgrootte = abs(doeltransform.a)
    tellers = np.zeros(doelvorm, dtype=np.int16)
    gemeten = 0
    verslag: list[dict] = []

    for laag in lagen:
        try:
            uitsnede = _uitsnede(laag, gebied, instellingen)
        except Exception as fout:
            logger.warning("Jaargang %s niet op te halen: %s", laag, fout)
            continue
        getal = _gelijk_raster(ndvi(uitsnede), uitsnede.transform, doelvorm, doeltransform)

        # 'Actueel' is een kopie van de nieuwste jaargang, dus die zou dezelfde opname twee
        # keer laten meetellen en de consensus scheef trekken.
        vingerafdruk = _vingerafdruk(getal)
        if any(abs(vingerafdruk - eerder) < 1e-9 for eerder in vingerafdrukken):
            logger.info("%s is dezelfde opname als een al gebruikte jaargang, overgeslagen", laag)
            continue
        vingerafdrukken.append(vingerafdruk)

        # Eerst de ligging gelijktrekken, dan pas vergelijken.
        if referentiebeeld is None:
            referentiebeeld = getal
            verplaatsing = (0.0, 0.0)
        else:
            verplaatsing = schat_verschuiving(referentiebeeld, getal)
            getal = _verschuif(getal, verplaatsing)

        # Daarna de kleurzweem, met een robuuste mediaanverschuiving.
        monster = getal[binnen] if binnen is not None and binnen.any() else getal
        mediaan = float(np.nanmedian(monster))
        if referentie_mediaan is None:
            referentie_mediaan = mediaan
        bijstelling = referentie_mediaan - mediaan
        begroeid = (getal + bijstelling) > drempel

        tellers += begroeid.astype(np.int16)
        gemeten += 1
        afstand_m = float(np.hypot(*verplaatsing) * pixelgrootte)
        verslag.append({
            "laag": laag,
            "mediaan_ndvi": round(mediaan, 4),
            "kleurbijstelling": round(bijstelling, 4),
            "verschuiving_m": round(afstand_m, 2),
            "aandeel_begroeid": round(float(begroeid[binnen].mean() if binnen is not None and binnen.any() else begroeid.mean()), 3),
        })
        logger.info(
            "%s: %.2f m verschoven, mediaan NDVI %+.4f bijgesteld met %+.4f, %.0f%% begroeid",
            laag, afstand_m, mediaan, bijstelling, 100 * verslag[-1]["aandeel_begroeid"],
        )
        if gemeten >= aantal_jaren:
            break

    if len(verslag) < 2:
        return None
    zekerheid = tellers.astype(np.float32) / max(gemeten, 1)
    return tellers, zekerheid, verslag
