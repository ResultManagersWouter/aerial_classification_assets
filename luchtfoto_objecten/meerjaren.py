"""Groen, bomen en verharding over meerdere jaargangen, en wat er veranderd is.

Drie dingen maken een vergelijking over jaren lastig, en die worden hier alle drie
aangepakt.

Elke jaargang heeft zijn eigen kleurzweem. Een vaste drempel die op 2026 klopt, mist
in 2023 de helft. Daarom wordt de drempel per jaargang opnieuw geijkt op dezelfde
registratie, zodat elk jaar op zijn eigen beste werkpunt staat. Wat je dan vergelijkt zijn
detecties die elk even goed zijn afgeregeld, en niet een getal dat meedrijft met het beeld.

Niet elke jaargang heeft beide opnamen. Van sommige jaren is er alleen kleur, van andere
alleen 25cm infrarood. Daarom wordt per jaar vastgelegd wat er is, en de vergelijking over
jaren gaat over de modaliteit die in álle gekozen jaren bestaat. Anders vergelijk je het
verschil tussen infrarood en kleur en noem je dat verandering.

En bomen wisselen met het seizoen. In volle bloei is de kroon veel groter dan in het
voorjaar, als de boom kaal is en je hem nauwelijks ziet. Kroonoppervlak van jaar tot jaar
naast elkaar leggen meet dus vooral de bladstand. Daarom krijgt elke jaargang een
bladstandscore, telt een boom als aanwezig zodra hij in één jaargang mét blad gevonden
wordt, en wordt verdwijnen alleen gemeld als de boom ontbreekt in een jaargang die wél
blad heeft.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

from luchtfoto_objecten.detectie.kenmerken import exces_groen
from luchtfoto_objecten.instellingen import Instellingen
from luchtfoto_objecten.kalibratie import ijk_drempel, referentiemaskers
from luchtfoto_objecten.modeldetectie import _kenmerken_van, naar_laag, opschonen
from luchtfoto_objecten.wmts import LuchtfotoWMTS, haal_beschikbare_lagen

logger = logging.getLogger(__name__)

JAARPATROON = re.compile(r"^(\d{4})_(quick)?ortho(HR|25)(IR)?$")
BLADGRENS = 0.15  # aandeel groene pixels waarboven we een opname als 'met blad' beschouwen


@dataclass
class Jaargang:
    jaar: int
    rgb: str | None = None
    infrarood: str | None = None
    bladaandeel: float = 0.0
    drempel: float = float("nan")
    iou: float = 0.0
    lagen: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)

    @property
    def heeft_blad(self) -> bool:
        return self.bladaandeel >= BLADGRENS

    @property
    def modaliteiten(self) -> set[str]:
        return {naam for naam, laag in (("rgb", self.rgb), ("infrarood", self.infrarood)) if laag}


def beschikbare_jaargangen(cache_map=None) -> dict[int, Jaargang]:
    """Per jaar de fijnste kleur- en infraroodlaag die PDOK heeft."""
    jaargangen: dict[int, Jaargang] = {}
    rang: dict[tuple[int, str], int] = {}
    for naam in haal_beschikbare_lagen(cache_map):
        treffer = JAARPATROON.match(naam)
        if not treffer:
            continue
        jaar = int(treffer.group(1))
        soort = "infrarood" if treffer.group(4) else "rgb"
        # 8cm gaat voor 25cm, en een definitieve opname voor een quick-versie.
        fijnheid = (0 if treffer.group(3) == "HR" else 1, 1 if treffer.group(2) else 0)
        jaargang = jaargangen.setdefault(jaar, Jaargang(jaar=jaar))
        if (jaar, soort) not in rang or fijnheid < rang[(jaar, soort)]:
            rang[(jaar, soort)] = fijnheid
            setattr(jaargang, soort, naam)
    return dict(sorted(jaargangen.items(), reverse=True))


def _uitsnede(laag: str, gebied, instellingen: Instellingen, zoom: int | None = None):
    from luchtfoto_objecten.bladstand import _passende_zoom

    gekozen = zoom or _passende_zoom(laag, instellingen.luchtfoto.zoom)
    wmts = LuchtfotoWMTS(
        laag=laag, zoom=gekozen, cache_map=instellingen.cache_map,
        max_werkers=instellingen.luchtfoto.max_werkers,
    )
    return wmts.haal_uitsnede(gebied.bbox, toon_voortgang=False)


def _vegetatiegetal(uitsnede, soort: str) -> np.ndarray:
    """Het getal waarop vegetatie herkend wordt: NDVI op infrarood, excess green op kleur."""
    if soort == "infrarood":
        from luchtfoto_objecten.infrarood import ndvi

        return ndvi(uitsnede)
    return exces_groen(uitsnede.afbeelding)


def verwerk_jaargang(
    jaargang: Jaargang, soort: str, gebied, registratie, analysevlak, instellingen: Instellingen
) -> Jaargang:
    """Detecteert groen en verharding op één jaargang, met een op de registratie geijkte drempel."""
    laag = getattr(jaargang, soort)
    if laag is None:
        return jaargang
    uitsnede = _uitsnede(laag, gebied, instellingen)
    kenmerken = _kenmerken_van(uitsnede, registratie, analysevlak, instellingen)
    jaargang.bladaandeel = float((exces_groen(uitsnede.afbeelding) > 0.05).mean())

    getal = _vegetatiegetal(uitsnede, soort)
    groen_ref, verhard_ref = referentiemaskers(registratie, kenmerken.vorm, uitsnede.transform, kenmerken.maaiveld)
    keuze = ijk_drempel(getal, groen_ref, verhard_ref)
    if not keuze.bruikbaar or not np.isfinite(keuze.drempel):
        logger.warning("Jaargang %s (%s): niet te ijken, overgeslagen", jaargang.jaar, laag)
        return jaargang
    jaargang.drempel, jaargang.iou = keuze.drempel, keuze.iou
    logger.info(
        "%s %s: drempel %.4f, IoU %.3f, AUC %.3f, bladaandeel %.0f%%",
        jaargang.jaar, laag, keuze.drempel, keuze.iou, keuze.scheidend_vermogen, jaargang.bladaandeel * 100,
    )

    groen = (getal > keuze.drempel) & kenmerken.maaiveld
    verharding = kenmerken.maaiveld & ~groen
    jaargang.lagen = {
        f"groen_{jaargang.jaar}": naar_laag(
            opschonen(groen, uitsnede.transform, instellingen.groen.min_oppervlakte_m2),
            uitsnede.transform, "groen", str(jaargang.jaar),
            instellingen.groen.min_oppervlakte_m2, instellingen.groen.vereenvoudiging_m, analysevlak,
        ),
        f"verharding_{jaargang.jaar}": naar_laag(
            opschonen(verharding, uitsnede.transform, instellingen.verharding.min_oppervlakte_m2),
            uitsnede.transform, "verharding", str(jaargang.jaar),
            instellingen.verharding.min_oppervlakte_m2, instellingen.verharding.vereenvoudiging_m, analysevlak,
        ),
    }
    return jaargang


def kies_modaliteit(jaargangen: list[Jaargang]) -> str | None:
    """De opnamesoort die álle gekozen jaren hebben, anders vergelijk je appels met peren."""
    if not jaargangen:
        return None
    gedeeld = set.intersection(*(j.modaliteiten for j in jaargangen))
    for voorkeur in ("infrarood", "rgb"):
        if voorkeur in gedeeld:
            return voorkeur
    return None


def verschillen(jaargangen: list[Jaargang], klasse: str) -> pd.DataFrame:
    """Wat er tussen opeenvolgende jaren bij komt en af gaat."""
    op_jaar = sorted((j for j in jaargangen if j.lagen), key=lambda j: j.jaar)
    regels = []
    for vorig, huidig in zip(op_jaar, op_jaar[1:]):
        eerder = vorig.lagen.get(f"{klasse}_{vorig.jaar}")
        later = huidig.lagen.get(f"{klasse}_{huidig.jaar}")
        if eerder is None or later is None or eerder.empty or later.empty:
            continue
        oud, nieuw = eerder.geometry.union_all(), later.geometry.union_all()
        regels.append({
            "klasse": klasse,
            "van": vorig.jaar,
            "naar": huidig.jaar,
            "oppervlakte_van_m2": round(oud.area, 1),
            "oppervlakte_naar_m2": round(nieuw.area, 1),
            "erbij_m2": round(nieuw.difference(oud).area, 1),
            "eraf_m2": round(oud.difference(nieuw).area, 1),
            "bladaandeel_van": round(vorig.bladaandeel, 3),
            "bladaandeel_naar": round(huidig.bladaandeel, 3),
        })
    return pd.DataFrame(regels)


def per_asset(jaargangen: list[Jaargang], registratie, bron: str, klasse: str) -> pd.DataFrame:
    """Per geregistreerd object hoeveel ervan elk jaar op de foto terug te zien is."""
    objecten = registratie.get(bron)
    if objecten is None or objecten.empty:
        return pd.DataFrame()

    rijen = []
    detecties = {
        j.jaar: j.lagen[f"{klasse}_{j.jaar}"].geometry.union_all()
        for j in jaargangen
        if j.lagen.get(f"{klasse}_{j.jaar}") is not None and not j.lagen[f"{klasse}_{j.jaar}"].empty
    }
    for positie, rij in objecten.iterrows():
        geometrie = rij.geometry
        if geometrie is None or geometrie.is_empty or geometrie.area < 2.0:
            continue
        regel = {
            "bron": bron,
            "identificatie": str(rij.get("identificatie", rij.get("id", positie))),
            "oppervlakte_m2": round(geometrie.area, 1),
        }
        for jaar, vlak in sorted(detecties.items()):
            regel[f"dekking_{jaar}"] = round(geometrie.intersection(vlak).area / geometrie.area, 3)
        dekkingen = [waarde for sleutel, waarde in regel.items() if sleutel.startswith("dekking_")]
        if len(dekkingen) >= 2:
            regel["verandering"] = round(dekkingen[-1] - dekkingen[0], 3)
        rijen.append(regel)
    return pd.DataFrame(rijen)


def analyseer(gebied, instellingen: Instellingen | None = None, aantal_jaren: int = 4):
    """Draait de detectie op de laatste jaargangen en vergelijkt ze onderling."""
    from luchtfoto_objecten.modeldetectie import haal_beelden

    instellingen = instellingen or Instellingen.laden()
    _, _, registratie, analysevlak = haal_beelden(gebied, instellingen)

    beschikbaar = list(beschikbare_jaargangen(instellingen.cache_map).values())[: aantal_jaren + 2]
    soort = kies_modaliteit(beschikbaar[:aantal_jaren]) or "rgb"
    gekozen = [j for j in beschikbaar if getattr(j, soort)][:aantal_jaren]
    logger.info(
        "Vergelijking over %s op %s: %s",
        soort, ", ".join(str(j.jaar) for j in gekozen), ", ".join(getattr(j, soort) for j in gekozen),
    )

    verwerkt = [verwerk_jaargang(j, soort, gebied, registratie, analysevlak, instellingen) for j in gekozen]
    verwerkt = [j for j in verwerkt if j.lagen]

    overzicht = pd.DataFrame([{
        "jaar": j.jaar,
        "opname": getattr(j, soort),
        "modaliteit": soort,
        "bladaandeel": round(j.bladaandeel, 3),
        "met_blad": j.heeft_blad,
        "drempel": round(j.drempel, 4),
        "iou_tegen_registratie": round(j.iou, 3),
        "groen_m2": round(float(j.lagen[f"groen_{j.jaar}"].geometry.area.sum()), 1),
        "verharding_m2": round(float(j.lagen[f"verharding_{j.jaar}"].geometry.area.sum()), 1),
    } for j in verwerkt])

    lagen = {naam: laag for j in verwerkt for naam, laag in j.lagen.items()}
    veranderingen = pd.concat(
        [verschillen(verwerkt, "groen"), verschillen(verwerkt, "verharding")], ignore_index=True
    )
    assets = pd.concat(
        [
            per_asset(verwerkt, registratie, "groenobjecten", "groen"),
            per_asset(verwerkt, registratie, "verhardingen", "verharding"),
        ],
        ignore_index=True,
    )
    return lagen, overzicht, veranderingen, assets
