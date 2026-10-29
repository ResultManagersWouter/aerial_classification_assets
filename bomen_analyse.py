"""Bomen in Amsterdam: staat elke geregistreerde boom er echt, en maar één keer?

Voor elk boompunt in de bomenkaart (bomen-WFS, alles behalve stobbes) zoekt deze analyse
uit of er op die plek een boom te vinden is, of er misschien twee punten op één boom staan,
of dat er helemaal niets staat. Het sterkst is de uitspraak bij solitaire bomen: een boom
langs de straat of los in het gras heeft een eigen kroon, en daar horen geen twee punten in.
In bos en dichte groepen groeien kronen in elkaar en is van bovenaf niet te zeggen welk
punt bij welke stam hoort.

Daarom twee lagen in de uitvoer:

    solitaire_bomen   geen andere geregistreerde boom binnen 5 m en niet in bos;
                      hier is het oordeel betrouwbaar
    alle_bomen        alle boompunten, met per boom hoe zeker het oordeel is

De bronnen:

    kronen        boomtoppen en kroonvlakken uit AHN5 (Amsterdam rond voorjaar 2023) en
                  AHN4 (2020-2022), zie luchtfoto_objecten/boomkronen.py
    vegetatie     NDVI uit de infraroodluchtfoto van de zomer van 2025 (blad aan de bomen)
                  en het voorjaar van 2026, het meest recente beeld
    registratie   de andere boompunten: staat er een tweede punt binnen 1 m, of in
                  dezelfde kroon?
    straatbeeld   panorama van Amsterdam en Google Street View bij alles wat aandacht vraagt

Mogelijke uitkomsten:

    Boom aanwezig                         eigen kroon in AHN5, top dichtbij het punt
    Boom aanwezig, kroon gedeeld          in een rij of groep waar kronen samengroeien
    Waarschijnlijk dubbel                 een tweede boompunt binnen 1 m
    Mogelijk dubbel                       tweede punt binnen 2,5 m in dezelfde kroon, terwijl
                                          er maar één top is (alleen bij solitaire bomen)
    Kroon op afstand                      geen kroon op het punt, wel een kroon zonder eigen
                                          punt op 1,5 tot 5 m: mogelijk verkeerd ingemeten
    Jonge boom, gezien                    geplant na het AHN, wel groen op de luchtfoto van 2025;
                                          staat er in AHN5 een kroon, dan is dat de vorige boom
    Jonge boom, niet te zien              geplant na het AHN, niet zichtbaar: straatbeeld
    Kleine boom, gezien                   geen kroon in AHN5, wel groen op de luchtfoto van 2025
    Kroon uit 2023 niet meer groen        eigen kroon in AHN5, maar geen groen in 2025 en 2026:
                                          gekapt, vervangen door een jonge boom, of kwijnend.
                                          Steekproef op straatbeeld: allebei de soorten komen voor
    Onder de kroon van een buurboom       geen eigen top, wel onder een kroon die bij een ander
                                          punt hoort; een kleine boom onder een grote is van
                                          bovenaf niet te zien
    Geen boom gevonden                    geen kroon en geen groen, en oud genoeg om te zien
    Niet te beoordelen                    in bos of een dichte groep zonder eigen kroon

Hoe goed de methode werkt, staat in het rapport: getoetst op oude straatbomen die er
zeker staan, en op stobbes waar de kap en een inspectie zonder boom zijn vastgelegd.

Draaien gaat via bomen_main.py.
"""

from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from luchtfoto_objecten import bomenregister, boomkronen, opslag, puntmeting, straatbeeld
from luchtfoto_objecten.geo_hulp import maak_schrijfbaar
from luchtfoto_objecten.uitvoer import tabel_md
import stobbes_controle

logger = logging.getLogger("bomen_analyse")

UITVOER_MAP = Path("output/bomen_analyse")
ONEDRIVE_BOMEN = stobbes_controle.ONEDRIVE_DATA

ZOMER, VOORJAAR = "2025_ortho25IR", "2026_orthoHRIR"
DUBBEL_EXACT_M = 0.3         # twee punten hierbinnen zijn in elke omgeving dezelfde boom
DUBBEL_ZEKER_M = 1.0
DUBBEL_MOGELIJK_M = 2.5
TOP_BIJ_PUNT_M = 3.0
KROON_OP_AFSTAND_M = (1.5, 5.0)
JONG_VANAF = 2019            # aangeplant vanaf dit jaar: in 2023 vaak nog te klein voor de kronen in AHN5
SMALLE_KROON = r"Fastigia|Columnar|Chanticleer|Ginkgo|Italica|Pyramidal|Erecta|Koster|Monument|Obelisk|Stricta"
AHN5_VLUCHT = pd.Timestamp("2023-04-01")  # later in de registratie gezet: mogelijk na het AHN geplant

STATUS = {
    "aanwezig": "Boom aanwezig",
    "aanwezig_gedeeld": "Boom aanwezig, kroon gedeeld",
    "dubbel_zeker": "Waarschijnlijk dubbel: twee punten op één boom",
    "dubbel_mogelijk": "Mogelijk dubbel: één kroon, twee punten",
    "kroon_op_afstand": "Kroon op afstand: mogelijk verkeerd ingemeten",
    "jong_gezien": "Jonge boom, gezien op luchtfoto 2025",
    "jong_niet_gezien": "Jonge boom, niet te zien: straatbeeld bekijken",
    "klein_gezien": "Kleine boom: groen op luchtfoto 2025, te laag voor het AHN",
    "verdwenen_na_2023": "Kroon uit 2023 niet meer groen: gekapt, vervangen of kwijnend",
    "onder_buurkroon": "Onder de kroon van een buurboom: niet apart te zien",
    "niet_gevonden": "Geen boom gevonden",
    "niet_te_beoordelen": "Niet te beoordelen (bos of dichte groep)",
}
ADVIES = {
    "aanwezig": "Geen actie",
    "aanwezig_gedeeld": "Geen actie",
    "dubbel_zeker": "Eén van de twee punten beëindigen; kijk welke de juiste gegevens heeft",
    "dubbel_mogelijk": "Straatbeeld bekijken: één boom of twee?",
    "kroon_op_afstand": "Straatbeeld bekijken en het punt naar de stam verplaatsen",
    "jong_gezien": "Geen actie",
    "jong_niet_gezien": "Straatbeeld bekijken: is de jonge boom geplant?",
    "klein_gezien": "Geen actie",
    "verdwenen_na_2023": ("Straatbeeld bekijken: weg (stobbe of beëindigen), vervangen door een jonge boom (plantjaar "
                          "en soort bijwerken) of kwijnend (conditie laten beoordelen)"),
    "onder_buurkroon": "Geen actie, tenzij er twijfel is; van bovenaf niet te scheiden",
    "niet_gevonden": "Straatbeeld bekijken; staat er niets, dan het punt beëindigen of verplaatsen",
    "niet_te_beoordelen": "Alleen bij twijfel in het veld controleren",
}
KLEUR = {
    "aanwezig": "#1baf7a", "aanwezig_gedeeld": "#7fd1ae", "dubbel_zeker": "#e34948", "dubbel_mogelijk": "#e87ba4",
    "kroon_op_afstand": "#6250d6", "jong_gezien": "#2a78d6", "jong_niet_gezien": "#9fc3ef", "klein_gezien": "#5aa0e6",
    "verdwenen_na_2023": "#eda100", "onder_buurkroon": "#a8d5c2", "niet_gevonden": "#eb6834",
    "niet_te_beoordelen": "#c9c8c2",
}
CONTROLEREN = {"dubbel_zeker", "dubbel_mogelijk", "kroon_op_afstand", "jong_niet_gezien", "verdwenen_na_2023",
               "niet_gevonden"}


# ------------------------------------------------------------------- meten


def dubbele_punten(bomen: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per boom het dichtstbijzijnde andere boompunt, voor de dubbelcontrole."""
    boom = cKDTree(np.c_[bomen.geometry.x, bomen.geometry.y])
    afstand, index = boom.query(np.c_[bomen.geometry.x, bomen.geometry.y], k=3)
    # Bij twee punten op exact dezelfde plek kan het punt zelf als tweede terugkomen;
    # neem dan de eerste buur die niet het punt zelf is
    zelf = np.arange(len(bomen))
    kolom = np.where(index[:, 0] != zelf, 0, 1)
    rij = np.arange(len(bomen))
    return pd.DataFrame({
        "punt_id": bomen["punt_id"].values,
        "ander_punt_id": bomen["id"].values[index[rij, kolom]],
        "ander_punt_afstand_m": np.round(afstand[rij, kolom], 2),
    })


def kronen(bomen: gpd.GeoDataFrame) -> pd.DataFrame:
    """AHN5- en AHN4-kroon per boompunt, en hoeveel boompunten dezelfde kroon delen."""
    tegels = opslag.kilometertegels(bomen.geometry.x, bomen.geometry.y)
    tabel = bomen[["punt_id"]].copy()
    for versie in (5, 4):
        toppen = boomkronen.zorg_voor_kronen(versie, tegels)
        bij = boomkronen.kronen_bij_punten(versie, bomen, toppen)
        p = f"ahn{versie}_"
        aantal = bij.groupby(p + "kroon_id").size()
        bij[p + "punten_in_kroon"] = bij[p + "kroon_id"].map(aantal).fillna(0).astype(int)
        if versie == 5:
            # Kronen die van geen enkel boompunt zijn: kandidaat voor 'kroon op afstand'
            geclaimd = set(bij[p + "kroon_id"].dropna())
            vrij = toppen[~toppen["kroon_id"].isin(geclaimd)].drop_duplicates("kroon_id")
            vrije_toppen = cKDTree(vrij[["x", "y"]].values)
            afstand, index = vrije_toppen.query(np.c_[bomen.geometry.x, bomen.geometry.y])
            bij["vrije_kroon_afstand_m"] = np.round(afstand, 2)
            bij["vrije_kroon_hoogte_m"] = vrij["hoogte_m"].values[index]
        tabel = tabel.merge(bij, on="punt_id", how="left")
    return tabel


def vegetatie(punten: gpd.GeoDataFrame, doel: str, opnieuw: bool = False) -> pd.DataFrame:
    tabel = punten[["punt_id"]].copy()
    for laag in (ZOMER, VOORJAAR):
        gemeten = puntmeting.bewaard(doel, laag, punten, lambda p, laag=laag: puntmeting.meet_vegetatiegetal(p, laag),
                                     opnieuw, kolommen=["ndvi_p90_kroon"])
        tabel = tabel.merge(gemeten[["punt_id", "ndvi_kern", "ndvi_contrast", "ndvi_p90_kroon"]]
                            .rename(columns={"ndvi_kern": f"ndvi_{laag}", "ndvi_contrast": f"ndvi_contrast_{laag}",
                                             "ndvi_p90_kroon": f"ndvi_p90_{laag}"}),
                            on="punt_id", how="left")
    return tabel


def groendrempels(bomen: pd.DataFrame, kale_plekken: pd.DataFrame) -> dict[str, tuple[str, float, float]]:
    """Per opname de NDVI-drempel die een boom van een gekapte plek scheidt (Youden).

    Boom: solitaire straatbomen van voor 2015 met een eigen AHN5-kroon van minstens 5 m.
    Geen boom: stobbes op straat, gekapt voor de vlucht van díe opname en daarna door een
    inspecteur zonder boom gezien, zonder levende boom binnen 3 m. Per opname dus een eigen
    set: voor de zomer van 2025 telt alles wat voor mei 2025 gekapt is.

    Twee maten worden vergeleken: de mediaan binnen 1,5 m van de stam, en het groenste deel
    (90e percentiel) binnen 3 m. Een kroon helt op de luchtfoto weg van de stam, dus de
    tweede kan beter zijn, maar vangt ook eerder de kroon van een buur. Per opname wint de
    maat met de hoogste AUC. Geeft per opname (maat, drempel, auc).
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    drempels = {}
    for laag in (ZOMER, VOORJAAR):
        beste = ("ndvi", 0.0, float("nan"))
        for maat in ("ndvi", "ndvi_p90"):
            ja = bomen.loc[bomen["referentie_boom"], f"{maat}_{laag}"].dropna()
            nee = kale_plekken.loc[kale_plekken[f"kaal_voor_{laag}"], f"{maat}_{laag}"].dropna()
            if len(ja) < 50 or len(nee) < 50:
                logger.warning("%s: te weinig ijkpunten (%s boom, %s kaal)", laag, len(ja), len(nee))
                continue
            waar, score = np.r_[np.ones(len(ja)), np.zeros(len(nee))], np.r_[ja, nee]
            auc = roc_auc_score(waar, score)
            if np.isnan(beste[2]) or auc > beste[2]:
                fpr, tpr, grenzen = roc_curve(waar, score)
                beste = (maat, float(grenzen[np.argmax(tpr - fpr)]), round(auc, 3))
        drempels[laag] = beste
    return drempels


# ------------------------------------------------------------------- oordeel


def beoordeel(t: pd.DataFrame, drempels: dict[str, tuple[str, float, float]]) -> pd.DataFrame:
    t = t.copy()
    (maat_z, drempel_z, _), (maat_v, drempel_v, _) = drempels[ZOMER], drempels[VOORJAAR]
    groen_zomer = t[f"{maat_z}_{ZOMER}"].gt(drempel_z)
    groen_voorjaar = t[f"{maat_v}_{VOORJAAR}"].gt(drempel_v)
    t["groen_2025"] = groen_zomer.astype("boolean").mask(t[f"{maat_z}_{ZOMER}"].isna())
    t["groen_2026"] = groen_voorjaar.astype("boolean").mask(t[f"{maat_v}_{VOORJAAR}"].isna())
    eigen_kroon = t["ahn5_kroon_id"].notna() & t["ahn5_afstand_top_m"].le(TOP_BIJ_PUNT_M)
    rand_kroon = t["ahn5_kroon_id"].notna() & ~eigen_kroon      # onder een kroon met de top verder weg
    kroon4 = t["ahn4_kroon_id"].notna()
    # Jong: geplant vanaf 2022. Zonder plantjaar alleen als het punt na de AHN5-vlucht in de
    # registratie kwam en er in 2023 nog geen eigen kroon van 5 m stond; anders is het een
    # bestaande boom die later is ingevoerd (gemeten: die hebben in AHN4 en AHN5 al 6-16 m)
    jong = t["jaar_van_aanleg"].fillna(0).ge(JONG_VANAF) | (
        t["jaar_van_aanleg"].isna() & pd.to_datetime(t["geldig_vanaf"], errors="coerce").ge(AHN5_VLUCHT)
        & ~(eigen_kroon & t["ahn5_kroon_hoogte_m"].ge(5)))
    verhard = t["omgeving"].eq("straat")
    bos = t["omgeving"].eq("bos")
    gedeeld = t["ahn5_punten_in_kroon"].gt(1)
    afstand_ander = t["ander_punt_afstand_m"]
    nergens_groen = ~groen_zomer.fillna(False) & ~groen_voorjaar.fillna(False) & t[f"ndvi_{ZOMER}"].notna()
    # Ook het groenste deel binnen 3 m moet kaal zijn: een smalle of ijle kroon helt op de
    # luchtfoto weg van de stam en lijkt bij de stam zelf kaal (steekproef op straatbeeld)
    kaal_op_straat = verhard & nergens_groen & ~t[f"ndvi_p90_{ZOMER}"].gt(drempel_z).fillna(False)

    regels = [
        # Onder 30 cm altijd dubbel; tot een meter alleen als er verder geen boom in de buurt
        # staat, want berken en bosplantsoen worden in groepjes dicht op elkaar geplant
        ("dubbel_zeker", afstand_ander.le(DUBBEL_EXACT_M)
         | (afstand_ander.le(DUBBEL_ZEKER_M) & ~bos & (t["bomen_binnen_5m"] <= 1))),
        ("dubbel_mogelijk", afstand_ander.le(DUBBEL_MOGELIJK_M) & gedeeld & ~bos & (t["bomen_binnen_5m"] <= 1)),
        ("dubbel_mogelijk", afstand_ander.le(DUBBEL_ZEKER_M)),
        # Een jonge boom is geplant na het AHN: een kroon in AHN5 is dan van de vorige boom
        # op die plek, dus voor hem telt alleen of hij in 2025 groen is. In gras is alles
        # groen, daar moet de plek groener zijn dan de omgeving
        ("jong_gezien", jong & groen_zomer.fillna(False) & (verhard | t[f"ndvi_contrast_{ZOMER}"].gt(0.05))),
        ("jong_niet_gezien", jong),
        ("aanwezig", eigen_kroon & ~gedeeld & ~kaal_op_straat),
        ("aanwezig_gedeeld", eigen_kroon & gedeeld & ~kaal_op_straat),
        ("verdwenen_na_2023", eigen_kroon & kaal_op_straat),
        ("onder_buurkroon", rand_kroon & gedeeld),
        ("aanwezig", rand_kroon & groen_zomer.fillna(False)),
        # Geen kroon in AHN5, wel groen op de plek: een kleine boom (dakplataan, leiboom, of
        # na 2023 flink gegroeid) die onder de 3 m van de toppenzoeker bleef
        ("klein_gezien", ~eigen_kroon & groen_zomer.fillna(False) & (verhard | t[f"ndvi_contrast_{ZOMER}"].gt(0.05))),
        ("kroon_op_afstand", ~eigen_kroon & t["vrije_kroon_afstand_m"].between(*KROON_OP_AFSTAND_M) & ~bos),
        ("niet_te_beoordelen", ~eigen_kroon & bos),
        ("niet_gevonden", ~eigen_kroon),
    ]
    sleutel = pd.Series("niet_te_beoordelen", index=t.index)
    for naam, masker in reversed(regels):
        sleutel[masker.fillna(False)] = naam
    t["status_sleutel"] = sleutel
    t["status"] = sleutel.map(STATUS)
    t["advies"] = sleutel.map(ADVIES)

    hoog = (
        # Na 2023 is alleen op verharding te zien of een kroon er nog is; in gras blijft het groen
        ((sleutel == "aanwezig") & eigen_kroon & verhard & groen_zomer.fillna(False))
        | (sleutel == "dubbel_zeker")
        | ((sleutel == "niet_gevonden") & t["solitair"] & ~groen_zomer.fillna(False) & ~kroon4 & t["ahn5_dichtste_top_m"].gt(5))
        | ((sleutel == "verdwenen_na_2023") & t["solitair"])
        | ((sleutel == "jong_gezien") & t["solitair"] & verhard)
    )
    # Smalle zuilvormen en gesnoeide vormbomen zijn van bovenaf slecht te zien: in de
    # steekproef op straatbeeld stonden een ginkgo, een zuilhaagbeuk en dakplatanen er gewoon
    lastig_van_boven = (t["soortnaam"].fillna("").str.contains(SMALLE_KROON, case=False)
                        | t["type_object"].isin(["Vormboom", "Knotboom", "Leiboom", "Gekandelaberde boom"]))
    hoog &= ~(lastig_van_boven & sleutel.isin(["niet_gevonden", "verdwenen_na_2023"]))
    laag = ((bos & sleutel.isin(["niet_gevonden", "kroon_op_afstand", "aanwezig_gedeeld", "dubbel_mogelijk"]))
            | sleutel.isin(["niet_te_beoordelen", "onder_buurkroon"])
            | (lastig_van_boven & sleutel.isin(["niet_gevonden", "verdwenen_na_2023"])))
    t["zekerheid"] = np.select([hoog, laag], ["hoog", "laag"], default="middel")

    volgorde = {"dubbel_zeker": 1, "niet_gevonden": 2, "verdwenen_na_2023": 3, "dubbel_mogelijk": 4,
                "kroon_op_afstand": 5, "jong_niet_gezien": 6, "onder_buurkroon": 7, "niet_te_beoordelen": 8,
                "klein_gezien": 9, "jong_gezien": 9, "aanwezig_gedeeld": 10, "aanwezig": 11}
    t["prioriteit"] = sleutel.map(volgorde) * 10 + t["zekerheid"].map({"hoog": 1, "middel": 2, "laag": 3})
    t["verdwenen_tussen_ahn4_ahn5"] = kroon4 & t["ahn5_kroon_id"].isna() & ~jong
    t["jong_op_plek_oude_boom"] = jong & eigen_kroon

    def bewijs(r) -> str:
        delen = []
        if pd.notna(r["ahn5_kroon_id"]):
            delen.append(f"AHN5-kroon {r['ahn5_kroon_hoogte_m']:.0f} m hoog, top op {r['ahn5_afstand_top_m']:.1f} m"
                         + (f", {int(r['ahn5_punten_in_kroon'])} punten in deze kroon" if r["ahn5_punten_in_kroon"] > 1 else ""))
        else:
            delen.append("geen AHN5-kroon" + (" (wel in AHN4)" if pd.notna(r["ahn4_kroon_id"]) else ""))
        if pd.notna(r[f"ndvi_{ZOMER}"]):
            delen.append(f"zomer 2025 {'groen' if r['groen_2025'] else 'niet groen'}")
        if r["ander_punt_afstand_m"] <= DUBBEL_MOGELIJK_M:
            delen.append(f"ander boompunt {int(r['ander_punt_id'])} op {r['ander_punt_afstand_m']:.1f} m")
        if r["status_sleutel"] == "kroon_op_afstand":
            delen.append(f"vrije kroon van {r['vrije_kroon_hoogte_m']:.0f} m op {r['vrije_kroon_afstand_m']:.1f} m")
        if pd.notna(r["jaar_van_aanleg"]) and r["jaar_van_aanleg"] > 0:
            delen.append(f"aanleg {int(r['jaar_van_aanleg'])}")
        return "; ".join(delen)

    t["bewijs"] = t.apply(bewijs, axis=1)
    return t


# ------------------------------------------------------------------- toetsen


def toets_methode(bomen: pd.DataFrame, gekapt: pd.DataFrame, drempels: dict[str, float]) -> pd.DataFrame:
    """Hoe vaak vindt de methode een boom waar er zeker een staat, en ten onrechte waar er zeker geen staat?

    Zeker wel: losstaande straatbomen van voor 2015 in hoogteklasse c of hoger.
    Zeker niet: stobbes gekapt voor maart 2023, daarna door een inspecteur zonder boom
    gezien, zonder levende boom binnen 3 m. Die gaan door precies dezelfde beoordeling als
    een gewoon boompunt, tussen alle echte bomen in.
    """
    gevonden = {"aanwezig", "dubbel_zeker", "dubbel_mogelijk"}
    van_buur = {"aanwezig_gedeeld", "onder_buurkroon"}
    rijen = []
    oud = bomen[bomen["solitair"] & bomen["omgeving"].eq("straat") & bomen["jaar_van_aanleg"].between(1900, 2014)
                & bomen["boomhoogteklasse_actueel"].str.match(r"^[c-g]\. ", na=False)]
    if len(oud):
        rijen.append({"toetsgroep": "oude losstaande straatboom (staat er zeker)", "punten": len(oud),
                      "pct_eigen_boom_gevonden": round(100 * oud["status_sleutel"].isin(gevonden).mean(), 1),
                      "pct_kroon_van_buur": round(100 * oud["status_sleutel"].isin(van_buur).mean(), 1),
                      "pct_gesignaleerd": round(100 * (~oud["status_sleutel"].isin(gevonden | van_buur)).mean(), 1)})
    for omgeving_ in ("straat", "groen", "bos"):
        sel = gekapt[gekapt["omgeving"].eq(omgeving_)] if len(gekapt) else gekapt
        if len(sel):
            rijen.append({"toetsgroep": f"gekapte plek, {omgeving_} (staat er zeker niet)", "punten": len(sel),
                          "pct_eigen_boom_gevonden": round(100 * sel["status_sleutel"].isin(gevonden).mean(), 1),
                          "pct_kroon_van_buur": round(100 * sel["status_sleutel"].isin(van_buur).mean(), 1),
                          "pct_gesignaleerd": round(100 * (~sel["status_sleutel"].isin(gevonden | van_buur)).mean(), 1)})
    for laag, (maat, drempel, auc) in drempels.items():
        naam = "groenste deel binnen 3 m" if maat == "ndvi_p90" else "mediaan bij de stam"
        rijen.append({"toetsgroep": f"groen {laag}: {naam}, drempel {drempel:.3f}, AUC {auc}".replace(".", ","),
                      "punten": None})
    return pd.DataFrame(rijen)


def schrijf_rapport(t: pd.DataFrame, toets: pd.DataFrame, pad: Path) -> Path:
    solitair = t[t["solitair"]]
    regels = [f"# Bomen in Amsterdam: staat elke boom er, en maar één keer? ({date.today():%d-%m-%Y})", ""]
    for kop, deel in (("Solitaire bomen (losstaand aan de straat of in het groen, niet in bos)", solitair), ("Alle bomen", t)):
        telling = deel.groupby(["status", "zekerheid"]).size().unstack(fill_value=0)
        telling["totaal"] = telling.sum(axis=1)
        telling = telling.reindex([STATUS[k] for k in STATUS if STATUS[k] in telling.index])
        regels += [f"## {kop}: {len(deel):,} boompunten".replace(f"{len(deel):,}", f"{len(deel):,}".replace(",", ".")), "",
                   tabel_md(telling.reset_index()), ""]
    regels += [
        "## Hoe goed werkt het?",
        "",
        "De methode is getoetst op punten waarvan het antwoord vaststaat. Een oude losstaande straatboom staat er "
        "vrijwel zeker; een stobbe die voor maart 2023 gekapt is en daarna door een inspecteur zonder boom is gezien, "
        "heeft vrijwel zeker geen boom. Beide gaan door dezelfde beoordeling, tussen alle echte bomen in.",
        "",
        "Lees de kolommen zo. *Eigen boom gevonden*: een eigen kroon (of een dubbel punt), dus 'boom aanwezig'. "
        "*Kroon van buur*: het punt valt in een kroon die ook bij een andere geregistreerde boom hoort; van bovenaf "
        "is dan niet te zeggen of er een eigen stam staat. *Gesignaleerd*: de methode zet het punt op de lijst om "
        "te controleren. Bij een boom die er zeker staat wil je veel 'eigen boom gevonden', bij een plek waar zeker "
        "niets staat veel 'gesignaleerd'.",
        "",
        tabel_md(toets) if len(toets) else "_In dit gebied te weinig zekere punten om de methode te toetsen; draai de hele stad._",
        "",
        "## Hoe het bepaald is",
        "",
        "Uit het AHN (hoogte boven maaiveld, 0,5 m) zijn alle boomtoppen gezocht en is per top de kroon afgebakend "
        "(lokale maxima boven 3 m, watershed tot 2 m). Elk boompunt is gekoppeld aan de kroon waarin het staat. "
        "Daarnaast is het vegetatiegetal (NDVI) gemeten op de infraroodluchtfoto van de zomer van 2025, met blad, "
        "en van het voorjaar van 2026. De drempel voor 'groen' is geijkt op bomen die er zeker staan tegen plekken "
        "waar zeker gekapt is.",
        "",
        "Een boompunt binnen 1 m van een ander boompunt is vrijwel altijd een dubbele registratie: straatbomen "
        "staan minstens vier meter uit elkaar. Binnen 2,5 m en in één kroon met één top is het bij een solitaire "
        "boom een sterke aanwijzing.",
        "",
        "## Grenzen",
        "",
        f"- Het AHN5 is van rond voorjaar 2023. Bomen die vanaf {JONG_VANAF} zijn aangeplant zijn te jong voor het "
        "AHN; voor hen telt alleen de luchtfoto, en een jonge kroon is daarop een paar pixels.",
        "- In bos en dichte groepen groeien kronen in elkaar; daar is van bovenaf niet te zeggen welk punt bij welke "
        "stam hoort. Die staan als 'niet te beoordelen' of met zekerheid laag.",
        "- 'Kroon uit 2023 niet meer groen' rust op het ontbreken van groen in 2025 en 2026 op een verharde plek. "
        "In een steekproef op straatbeeld stond er bij de ene boom een net geplante jonge boom (de oude was "
        "vervangen) en bij de andere een oude iep met een heel ijle kroon. Het signaal is dus: er is iets veranderd "
        "of mis met deze boom, niet per se: hij is weg.",
        "",
        "## Nalopen",
        "",
        "Open `bomen_analyse.gpkg` in QGIS. `te_controleren` bevat de solitaire bomen met een bevinding, op "
        "prioriteit, met per boom `panorama_beeld` (het nieuwste Amsterdamse panorama, gericht op de boom, met "
        "datum) en `google_streetview`. De laag `dubbele_punten` trekt een lijn tussen twee punten die waarschijnlijk "
        "dezelfde boom zijn.",
    ]
    pad.write_text("\n".join(regels) + "\n", encoding="utf-8")
    return pad


# ------------------------------------------------------------------- uitvoeren


def analyseer_bomen(bbox=None, opnieuw: bool = False, straatbeeld_ophalen: bool = True,
                    kopie_naar_onedrive: bool = True) -> gpd.GeoDataFrame:
    boompunten = bomenregister.alle_boompunten()
    stobbes, bomen = bomenregister.splits(boompunten)
    kap = bomenregister.kap_en_herplant()
    if bbox is not None:
        bomen = bomen.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True)
        stobbes = stobbes.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True)
        logger.info("Proef op %s: %s bomen", bbox, len(bomen))
    bomen["punt_id"] = puntmeting.punt_ids(bomen, "boom")
    stobbes["punt_id"] = puntmeting.punt_ids(stobbes, "stobbe")
    logger.info("%s boompunten", f"{len(bomen):,}")

    om = bomenregister.omgeving(bomen)
    om["punt_id"] = bomen["punt_id"].values
    t = (bomen.merge(om, on="punt_id").merge(dubbele_punten(bomen), on="punt_id")
         .merge(kronen(bomen), on="punt_id", how="left").merge(vegetatie(bomen, "bomen", opnieuw), on="punt_id", how="left"))

    # Toetsset 'zeker geen boom': stobbes gekapt voor de AHN5-vlucht, daarna zonder boom gezien
    traject = stobbes_controle.bekende_antwoorden(stobbes, kap)
    inspectie_nee = set(kap.loc[kap["boom_aanwezigheid"].eq("Nee"), "boom_id"].dropna())
    s = stobbes.merge(traject[["id", "kapdatum"]], on="id", how="left")
    boomindex = cKDTree(np.c_[bomen.geometry.x, bomen.geometry.y])
    buur, _ = boomindex.query(np.c_[s.geometry.x, s.geometry.y])
    # Kale plekken: gekapt, daarna door een inspecteur zonder boom gezien, en geen levende
    # boom binnen 3 m. Ze dienen om de groendrempel te ijken en om de methode te toetsen.
    kaal = s[s["kapdatum"].notna() & s["id"].isin(inspectie_nee) & (buur > 3.0)].reset_index(drop=True)
    gekapt = pd.DataFrame()
    if len(kaal):
        om_s = bomenregister.omgeving(kaal, bomen)
        om_s["punt_id"] = kaal["punt_id"].values
        kaal = (kaal.merge(om_s, on="punt_id").merge(vegetatie(kaal, "stobbes", opnieuw), on="punt_id", how="left")
                .merge(kronen(kaal), on="punt_id", how="left"))
        # Beoordelen alsof het een boompunt is tussen alle echte bomen: valt hij in een kroon
        # die bij een geregistreerde boom hoort, dan deelt hij die kroon
        bomen_per_kroon = t["ahn5_kroon_id"].value_counts()
        kaal["ahn5_punten_in_kroon"] = 1 + kaal["ahn5_kroon_id"].map(bomen_per_kroon).fillna(0).astype(int)
        afstand, index = boomindex.query(np.c_[kaal.geometry.x, kaal.geometry.y])
        kaal["ander_punt_afstand_m"] = np.round(afstand, 2)
        kaal["ander_punt_id"] = bomen["id"].values[index]
        for laag in (ZOMER, VOORJAAR):
            begin = pd.Timestamp(stobbes_controle.VLUCHT[laag][0])
            kaal[f"kaal_voor_{laag}"] = kaal["omgeving"].eq("straat") & kaal["kapdatum"].lt(begin)
        # Voor de toets alleen wat al voor de AHN5-vlucht weg was
        gekapt = kaal[kaal["kapdatum"].lt(AHN5_VLUCHT - pd.Timedelta(days=30))].reset_index(drop=True)
    t["referentie_boom"] = (t["solitair"] & t["omgeving"].eq("straat") & t["jaar_van_aanleg"].between(1900, 2014)
                            & t["ahn5_kroon_hoogte_m"].ge(5) & t["ahn5_punten_in_kroon"].eq(1))
    drempels = groendrempels(t, kaal) if len(kaal) else {ZOMER: ("ndvi", 0.0, np.nan), VOORJAAR: ("ndvi", 0.0, np.nan)}
    logger.info("NDVI-drempels voor groen: %s", drempels)

    t = beoordeel(t, drempels)
    if len(gekapt):
        gekapt = beoordeel(gekapt, drempels)
    toets = toets_methode(t, gekapt, drempels) if len(gekapt) else pd.DataFrame()

    # Straatbeeld: links voor iedereen, het panorama opzoeken alleen waar iets te controleren is
    controle = t["status_sleutel"].isin(CONTROLEREN) & (t["solitair"] | t["zekerheid"].ne("laag"))
    links = straatbeeld.links(bomen)
    if straatbeeld_ophalen and controle.any():
        pano = straatbeeld.panorama_bij_punten(bomen[controle.values], "bomen")
        links = links.merge(pano.drop(columns=["lat", "lon", "google_maps"]), on="punt_id", how="left")
    t = t.merge(links.drop(columns=["lat", "lon"]), on="punt_id", how="left")
    zonder = t["google_streetview"].isna()
    lon, lat = straatbeeld.NAAR_WGS84.transform(t.geometry.x.values, t.geometry.y.values)
    t.loc[zonder, "google_streetview"] = [straatbeeld.google_streetview(a, o) for a, o in zip(lat[zonder.values], lon[zonder.values])]

    UITVOER_MAP.mkdir(parents=True, exist_ok=True)
    pad = UITVOER_MAP / "bomen_analyse.gpkg"
    voorop = ["id", "status", "zekerheid", "advies", "prioriteit", "bewijs", "panorama_beeld", "panorama_datum",
              "google_streetview", "google_maps", "omgeving", "solitair", "soortnaam", "jaar_van_aanleg",
              "boomhoogteklasse_actueel", "type_object", "standplaats", "ander_punt_id", "ander_punt_afstand_m"]
    voorop = [k for k in voorop if k in t]
    meet = [k for k in t.columns if k.startswith(("ahn5_", "ahn4_", "ndvi_", "groen_", "vrije_kroon", "verdwenen_", "jong_"))]
    kolommen = voorop + meet + ["bomen_binnen_5m", "buurboom_afstand_m", "status_sleutel", "geometry"]
    uitvoer = gpd.GeoDataFrame(t[kolommen], geometry="geometry", crs="EPSG:28992").sort_values("prioriteit")
    if pad.exists():
        pad.unlink()
    maak_schrijfbaar(uitvoer[uitvoer["solitair"]]).to_file(pad, layer="solitaire_bomen", driver="GPKG", dataset_options={"VERSION": "1.2"})
    maak_schrijfbaar(uitvoer).to_file(pad, layer="alle_bomen", driver="GPKG")
    te_controleren = uitvoer[controle.reindex(uitvoer.index).fillna(False).values & uitvoer["solitair"].values]
    maak_schrijfbaar(te_controleren).to_file(pad, layer="te_controleren", driver="GPKG")

    paren = t[t["ander_punt_afstand_m"].le(DUBBEL_MOGELIJK_M)][["id", "ander_punt_id", "ander_punt_afstand_m", "status", "geometry"]]
    paren = paren[paren["id"] < paren["ander_punt_id"]]
    plek = bomen.set_index("id").geometry
    lijnen = gpd.GeoDataFrame(paren.drop(columns="geometry"),
                              geometry=[LineString([a, plek[b]]) for a, b in zip(paren.geometry, paren["ander_punt_id"])],
                              crs="EPSG:28992")
    maak_schrijfbaar(lijnen).to_file(pad, layer="dubbele_punten", driver="GPKG")

    for naam, tabel in {"overzicht_solitair": uitvoer[uitvoer["solitair"]].groupby(["status", "zekerheid"]).size().rename("bomen").reset_index(),
                        "overzicht_alle": uitvoer.groupby(["status", "zekerheid"]).size().rename("bomen").reset_index(),
                        "toets_methode": toets}.items():
        if len(tabel):
            pyogrio.write_dataframe(tabel, pad, layer=naam, driver="GPKG")
    rapport = schrijf_rapport(t, toets, UITVOER_MAP / "rapport_bomen.md")

    try:
        from luchtfoto_objecten.qgis_stijl import zet_stijl
        zet_stijl(pad, ["solitaire_bomen", "alle_bomen", "te_controleren"], "status", {STATUS[k]: KLEUR[k] for k in STATUS})
    except Exception as fout:  # de opmaak is een extraatje, de analyse staat er ook zonder
        logger.warning("QGIS-opmaak niet toegevoegd: %s", fout)

    print("\n" + rapport.read_text(encoding="utf-8").split("## Hoe het bepaald is")[0])
    print(f"GeoPackage: {pad}\nRapport:    {rapport}")
    opslag.werk_leesmij_bij()
    if kopie_naar_onedrive and bbox is None and ONEDRIVE_BOMEN.exists():
        stempel = f"{date.today():%d%m%Y}"
        shutil.copyfile(pad, ONEDRIVE_BOMEN / f"Bomen_analyse_{stempel}.gpkg")
        shutil.copyfile(rapport, ONEDRIVE_BOMEN / f"Bomen_analyse_{stempel}.md")
        print(f"Kopie in de datamap: {ONEDRIVE_BOMEN}")
    return uitvoer
