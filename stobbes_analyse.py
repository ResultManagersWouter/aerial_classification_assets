"""Stobbes in Amsterdam: welke zijn correct, en welke zijn al vervangen door een nieuwe boom?

Een stobbe is wat er na het kappen van een boom overblijft. Wordt er een nieuwe boom
geplant, dan hoort de stobbe uit de registratie te verdwijnen of een boom te worden. Dat
gaat niet altijd goed. Deze analyse zoekt per stobbe uit wat de open bronnen zeggen, en
legt vier soorten bewijs naast elkaar die elk iets anders zien:

    registratie   staat er een levende boom op dezelfde plek in de bomenkaart? Dan is de
                  herplant als nieuw punt geregistreerd en de stobbe niet afgevoerd.
    traject       het kap- en herplanttraject (bomen-WFS, laag kapenherplant): is er
                  gekapt, is er geplant, en wat zag de inspecteur bij de herplantinspectie?
    beeld         hoogte en kroon uit het AHN (2020-2023) en het vegetatiegetal uit de
                  infraroodluchtfoto's van 2023 tot en met 2026, zie stobbes_controle.py
    straatbeeld   het nieuwste panorama van Amsterdam en Google Street View, om zelf te kijken

De status per stobbe volgt uit die vier, in deze volgorde van zwaarte:

    Fout: stobbe en nieuwe boom op dezelfde plek
        een jonge levende boom staat binnen 1,5 m in de registratie, of het traject meldt herplant
    Fout: stobbe en bestaande boom op één plek geregistreerd
        een oude levende boom binnen 1,5 m: dezelfde boom staat er twee keer in; het nieuwste
        bewijs zegt welke van de twee weg moet
    Fout: herplant uitgevoerd, nieuwe boom niet geregistreerd
        het traject meldt planten of een nieuwe boom, en er staat geen boom binnen 3 m
    Mogelijk fout: er staat nog een boom
        geen kap uitgevoerd volgens het traject, en beeld of inspectie zien een boom
    Correct: geen boom
        gekapt of door de inspecteur zonder boom gezien, en het beeld spreekt dat niet tegen
    Waarschijnlijk correct: geen boom gezien
        geen traject, maar ook geen boom in beeld of registratie
    Onzeker: bekijk het straatbeeld
        de rest, meestal omdat het beeld twijfelt

De registratie en het traject zijn het sterkste bewijs: dat is wat de gemeente zelf heeft
vastgelegd. Het beeld is onafhankelijk maar heeft grenzen: een jonge boom is de eerste
jaren te klein voor het AHN en geeft op een luchtfoto van 21 cm nauwelijks een kroon, en
een stobbe in een bomenrij staat onder de kroon van zijn buren. Daarom is het straatbeeld
erbij gezet: met een klik zie je de plek vanaf de straat, met datum.

Draaien gaat via stobbe_main.py.
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

from luchtfoto_objecten import bomenregister, boomkronen, opslag, puntmeting, straatbeeld
from luchtfoto_objecten.geo_hulp import maak_schrijfbaar
from luchtfoto_objecten.uitvoer import tabel_md
import stobbes_controle

logger = logging.getLogger("stobbes_analyse")

UITVOER_MAP = Path("output/stobbes_analyse")
ONEDRIVE_BOMEN = stobbes_controle.ONEDRIVE_DATA

HERPLANT_OP_PLEK_M = 1.5     # een levende boom hierbinnen staat op de plek van de stobbe
HERPLANT_DICHTBIJ_M = 3.0    # hierbinnen is het vrijwel zeker de herplant, net iets verschoven
BEELD_BOOM = 0.65
BEELD_GEEN_BOOM = 0.35

STATUS = {
    "fout_zelfde_plek": "Fout: stobbe en nieuwe boom op dezelfde plek",
    "fout_dubbel": "Fout: stobbe en bestaande boom op één plek geregistreerd",
    "fout_niet_geregistreerd": "Fout: herplant uitgevoerd, nieuwe boom niet geregistreerd",
    "mogelijk_boom": "Mogelijk fout: er staat nog een boom",
    "correct": "Correct: geen boom",
    "waarschijnlijk_correct": "Waarschijnlijk correct: geen boom gezien",
    "onzeker": "Onzeker: bekijk het straatbeeld",
}
ADVIES = {
    "fout_zelfde_plek": "Stobbe beëindigen: de nieuwe boom staat al als eigen punt",
    "fout_dubbel": "Straatbeeld bekijken: staat de boom er, dan de stobbe beëindigen, anders het levende punt",
    "fout_niet_geregistreerd": "Nieuwe boom registreren en de stobbe beëindigen",
    "mogelijk_boom": "Straatbeeld bekijken; staat de boom er, het type terugzetten naar boom",
    "correct": "Geen actie",
    "waarschijnlijk_correct": "Geen actie; bij twijfel straatbeeld bekijken",
    "onzeker": "Straatbeeld bekijken en de registratie daarop aanpassen",
}
KLEUR = {  # voor de opmaak in QGIS
    "fout_zelfde_plek": "#e34948", "fout_dubbel": "#e87ba4", "fout_niet_geregistreerd": "#eb6834", "mogelijk_boom": "#eda100",
    "correct": "#1baf7a", "waarschijnlijk_correct": "#7fd1ae", "onzeker": "#6250d6",
}


# ------------------------------------------------------------------- bewijs


def buurbomen(stobbes: gpd.GeoDataFrame, bomen: gpd.GeoDataFrame) -> pd.DataFrame:
    """De dichtstbijzijnde levende boom per stobbe, en een tweede stobbe op dezelfde plek."""
    boom = cKDTree(np.c_[bomen.geometry.x, bomen.geometry.y])
    afstand, index = boom.query(np.c_[stobbes.geometry.x, stobbes.geometry.y])
    buur = bomen.iloc[index].reset_index(drop=True)
    uit = pd.DataFrame({
        "punt_id": stobbes["punt_id"].values,
        "buurboom_id": buur["id"].values,
        "buurboom_afstand_m": np.round(afstand, 2),
        "buurboom_type": buur["type_object"].values,
        "buurboom_soort": buur["soortnaam"].values,
        "buurboom_aanleg": buur["jaar_van_aanleg"].values,
        "buurboom_hoogteklasse": buur["boomhoogteklasse_actueel"].values,
        "buurboom_geldig_vanaf": pd.to_datetime(buur["geldig_vanaf"], errors="coerce").values,
    })
    eigen = cKDTree(np.c_[stobbes.geometry.x, stobbes.geometry.y])
    paren = eigen.query_ball_point(np.c_[stobbes.geometry.x, stobbes.geometry.y], 1.0)
    uit["andere_stobbe_binnen_1m"] = [len(p) > 1 for p in paren]
    return uit


def traject(stobbes: pd.DataFrame, kap: pd.DataFrame) -> pd.DataFrame:
    """Het laatste kap- en herplanttraject per stobbe, samengevat in wat ertoe doet."""
    bij = kap[kap["boom_id"].isin(stobbes["id"])].copy()
    bij["_volgorde"] = bij[["datum_herplantinspectie", "kapmaatregel_datum_uitgevoerd",
                            "plantmaatregel_datum_uitgevoerd"]].max(axis=1)
    laatste = bij.sort_values("_volgorde").drop_duplicates("boom_id", keep="last").set_index("boom_id")
    uit = pd.DataFrame({"id": stobbes["id"].values})
    kolommen = {
        "kapmaatregel_datum_uitgevoerd": "kapdatum",
        "werkgang_acceptatie_kap": "kap_acceptatie",
        "datum_herplantinspectie": "herplantinspectie_datum",
        "boom_aanwezigheid": "herplantinspectie_uitkomst",
        "plantlijst_bomen": "op_plantlijst",
        "plantmaatregel_datum_uitgevoerd": "plantdatum",
        "werkgang_acceptatie_planten": "plant_acceptatie",
        "deelopdracht_planten": "deelopdracht_planten",
        "toe_te_passen_boomsoort": "nieuwe_soort",
        "boom_nieuw_id": "boom_nieuw_id",
        "boomgebreken": "boomgebreken",
    }
    for bron, doel in kolommen.items():
        uit[doel] = uit["id"].map(laatste[bron]) if bron in laatste else np.nan
    uit["heeft_traject"] = uit["id"].isin(bij["boom_id"])
    uit["kap_uitgevoerd"] = uit["kapdatum"].notna() | uit["kap_acceptatie"].eq("Goed")
    uit["herplant_uitgevoerd"] = (uit["plantdatum"].notna() | uit["plant_acceptatie"].eq("Goed")
                                  | uit["herplantinspectie_uitkomst"].eq("Ja, nieuwe boom reeds aangeplant"))
    uit["herplant_status"] = np.select(
        [uit["herplant_uitgevoerd"], uit["op_plantlijst"].eq("Ja") & uit["heeft_traject"], uit["heeft_traject"]],
        ["uitgevoerd", "gepland, nog niet uitgevoerd", "geen herplant gepland"], default="geen traject bekend")
    return uit


def kroon_van_stobbe(stobbes: gpd.GeoDataFrame, bomen: gpd.GeoDataFrame) -> pd.DataFrame:
    """Staat de stobbe in een AHN5-kroon, en is dat de kroon van een geregistreerde buurboom?

    Een stobbe in een kroon die bij geen enkele levende boom hoort, had in 2023 een eigen
    kroon boven zich: toen stond daar dus nog een boom.
    """
    tegels = opslag.kilometertegels(stobbes.geometry.x, stobbes.geometry.y)
    toppen = boomkronen.zorg_voor_kronen(5, tegels)
    bij_stobbe = boomkronen.kronen_bij_punten(5, stobbes, toppen)

    # De levende bomen in de buurt van stobbes, om te zien welke kronen bij hen horen
    boom = cKDTree(np.c_[bomen.geometry.x, bomen.geometry.y])
    dichtbij = sorted(set().union(*boom.query_ball_point(np.c_[stobbes.geometry.x, stobbes.geometry.y], 15.0)))
    buren = bomen.iloc[dichtbij].copy()
    buren["punt_id"] = puntmeting.punt_ids(buren, "boom")
    bij_buren = boomkronen.kronen_bij_punten(5, buren, toppen)
    kronen_met_boom = set(bij_buren["ahn5_kroon_id"].dropna())
    bij_stobbe["kroon_van_buurboom"] = bij_stobbe["ahn5_kroon_id"].isin(kronen_met_boom)
    bij_stobbe["eigen_kroon_ahn5"] = bij_stobbe["ahn5_kroon_id"].notna() & ~bij_stobbe["kroon_van_buurboom"]
    return bij_stobbe


# ------------------------------------------------------------------- oordeel


# Peildatum van elke luchtfoto voor de tijdlijn: midden van de vliegperiode
BEELD_DATUM = {2024: pd.Timestamp("2024-07-15"), 2025: pd.Timestamp("2025-07-15"), 2026: pd.Timestamp("2026-03-15")}


def tijdlijn(r) -> tuple[str | None, str | None, pd.Timestamp | None]:
    """Het nieuwste stuk bewijs over de oude boom: staat hij er nog (boom) of niet (geen boom)?

    Kap, herplantinspectie en de luchtfoto's van 2024, 2025 en 2026 hebben elk een datum. Het
    nieuwste stuk dat iets beslist, wint: een inspectie 'Nee' van september 2026 weegt zwaarder
    dan een luchtfoto van maart 2026 met nog een kroon, en een luchtfoto van 2025 zonder boom
    weegt zwaarder dan een inspectie 'Ja' uit 2024. Een kroon op de luchtfoto telt alleen als
    het de eigen kroon is (niet die van een geregistreerde buur) of als de stobbe aan de straat staat.
    """
    bewijs = []
    if r["kap_uitgevoerd"] and pd.notna(r["kapdatum"]):
        bewijs.append((r["kapdatum"], "geen boom", "kap"))
    uitkomst = r["herplantinspectie_uitkomst"]
    if pd.notna(r["herplantinspectie_datum"]) and uitkomst in ("Ja", "Nee", "Ja, nieuwe boom reeds aangeplant"):
        bewijs.append((r["herplantinspectie_datum"], "boom" if uitkomst == "Ja" else "geen boom", f"inspectie {uitkomst}"))
    eigen = bool(r.get("eigen_kroon_ahn5")) or r.get("omgeving") == "straat"
    for jaar, datum in BEELD_DATUM.items():
        kans = r.get(f"kans_boom_{jaar}")
        if pd.isna(kans):
            continue
        # Na een vastgelegde kap moet een luchtfoto heel zeker zijn om dat tegen te spreken;
        # een twijfelachtige kroon is daar meestal die van een buurboom
        na_kap = r["kap_uitgevoerd"] and pd.notna(r["kapdatum"]) and datum > r["kapdatum"]
        if kans >= (0.8 if na_kap else BEELD_BOOM) and eigen:
            bewijs.append((datum, "boom", f"luchtfoto {jaar} (kans {kans:.2f})"))
        elif kans <= BEELD_GEEN_BOOM:
            bewijs.append((datum, "geen boom", f"luchtfoto {jaar} (kans {kans:.2f})"))
    if not bewijs:
        return None, None, None
    datum, toestand, bron = max(bewijs, key=lambda b: b[0])
    return toestand, bron, datum


def beoordeel(tabel: pd.DataFrame) -> pd.DataFrame:
    """Status, zekerheid, advies, prioriteit en een leesbare samenvatting van het bewijs."""
    t = tabel.copy()
    for kolom in ("kap_uitgevoerd", "herplant_uitgevoerd", "eigen_kroon_ahn5"):
        t[kolom] = t[kolom].fillna(False).astype(bool)
    afstand = t["buurboom_afstand_m"]
    jonge_buur = t["buurboom_aanleg"].fillna(0).ge(2018) | t["buurboom_hoogteklasse"].fillna("").str.startswith("a.")
    kans = t["kans_boom"]

    lijn = t.apply(tijdlijn, axis=1, result_type="expand")
    lijn.columns = ["toestand", "laatste_bron", "laatste_datum"]
    t = pd.concat([t, lijn], axis=1)
    boom_nu = t["toestand"].eq("boom")
    geen_nu = t["toestand"].eq("geen boom")
    uit_registratie = t["laatste_bron"].fillna("").str.match(r"^(kap|inspectie)")
    ooit_gekapt_of_nee = t["kap_uitgevoerd"] | t["herplantinspectie_uitkomst"].eq("Nee")

    # Een levende boom op de plek van de stobbe is een herplant als hij jong is of als het
    # traject een herplant meldt; is hij oud, dan is het dezelfde boom die twee keer staat
    regels = [
        ("fout_zelfde_plek", afstand.le(HERPLANT_OP_PLEK_M) & (jonge_buur | t["herplant_uitgevoerd"])),
        ("fout_dubbel", afstand.le(HERPLANT_OP_PLEK_M)),
        ("fout_niet_geregistreerd", t["herplant_uitgevoerd"] & afstand.gt(HERPLANT_DICHTBIJ_M)),
        ("fout_zelfde_plek", t["herplant_uitgevoerd"] & afstand.le(HERPLANT_DICHTBIJ_M)),
        ("mogelijk_boom", boom_nu),
        ("correct", geen_nu & (uit_registratie | ooit_gekapt_of_nee)),
        ("waarschijnlijk_correct", geen_nu),
    ]
    sleutel = pd.Series("onzeker", index=t.index)
    for naam, masker in reversed(regels):
        sleutel[masker.fillna(False)] = naam
    t["status_sleutel"] = sleutel
    t["status"] = sleutel.map(STATUS)
    t["advies"] = sleutel.map(ADVIES)
    # Staat de oude boom er nog en ligt er een oudere levende boom vlak naast de stobbe, dan
    # is dat waarschijnlijk dezelfde boom die twee keer geregistreerd is
    buur_id = t["buurboom_id"].astype("Int64").astype(str)
    t.loc[(sleutel == "fout_dubbel") & boom_nu, "advies"] = "De boom staat er nog: stobbe beëindigen"
    t.loc[(sleutel == "fout_dubbel") & geen_nu, "advies"] = (
        "De boom is weg: levende boom " + buur_id[(sleutel == "fout_dubbel") & geen_nu] + " beëindigen, stobbe houden")
    zelfde_boom = (sleutel == "mogelijk_boom") & afstand.le(HERPLANT_DICHTBIJ_M) & ~jonge_buur & ~t["kap_uitgevoerd"]
    t.loc[zelfde_boom, "advies"] = ("De boom staat er waarschijnlijk nog en is ook als levende boom "
                                    + t.loc[zelfde_boom, "buurboom_id"].astype("Int64").astype(str)
                                    + " geregistreerd: stobbe beëindigen")

    beeld_na_laatste = pd.Series(False, index=t.index)
    for jaar, datum in BEELD_DATUM.items():
        beeld_na_laatste |= t[f"kans_boom_{jaar}"].ge(BEELD_BOOM) & t["laatste_datum"].lt(datum)
    hoog = (
        ((sleutel == "fout_zelfde_plek") & (afstand.le(0.5) | t["herplant_uitgevoerd"]) & t["omgeving"].ne("bos"))
        | ((sleutel == "fout_dubbel") & afstand.le(0.5) & t["toestand"].notna())
        | ((sleutel == "fout_niet_geregistreerd") & t["plantdatum"].notna() & t["plant_acceptatie"].eq("Goed"))
        | ((sleutel == "correct") & uit_registratie & kans.le(BEELD_GEEN_BOOM).fillna(True) & ~beeld_na_laatste)
    )
    laag = ((sleutel == "onzeker") | ((sleutel == "mogelijk_boom") & ~uit_registratie & kans.lt(0.8))
            | (sleutel.isin(["fout_zelfde_plek", "fout_dubbel"]) & t["omgeving"].eq("bos") & afstand.gt(1.0)))
    t["zekerheid"] = np.select([hoog, laag], ["hoog", "laag"], default="middel")

    volgorde = {"fout_zelfde_plek": 1, "fout_dubbel": 2, "fout_niet_geregistreerd": 3, "mogelijk_boom": 4,
                "onzeker": 5, "waarschijnlijk_correct": 6, "correct": 7}
    t["prioriteit"] = sleutel.map(volgorde) * 10 + t["zekerheid"].map({"hoog": 1, "middel": 2, "laag": 3})

    def bewijs(r) -> str:
        delen = []
        if pd.notna(r["buurboom_afstand_m"]) and r["buurboom_afstand_m"] <= HERPLANT_DICHTBIJ_M:
            aanleg = f", aanleg {int(r['buurboom_aanleg'])}" if pd.notna(r["buurboom_aanleg"]) and r["buurboom_aanleg"] > 0 else ""
            delen.append(f"levende boom {int(r['buurboom_id'])} op {r['buurboom_afstand_m']:.1f} m{aanleg}")
        if r["kap_uitgevoerd"]:
            delen.append("gekapt" + (f" {r['kapdatum']:%Y-%m-%d}" if pd.notna(r["kapdatum"]) else ""))
        if r["herplant_uitgevoerd"]:
            delen.append("herplant" + (f" {r['plantdatum']:%Y-%m-%d}" if pd.notna(r["plantdatum"]) else " gemeld"))
        if pd.notna(r["herplantinspectie_uitkomst"]):
            datum = f" ({r['herplantinspectie_datum']:%Y-%m-%d})" if pd.notna(r["herplantinspectie_datum"]) else ""
            delen.append(f"inspectie: {r['herplantinspectie_uitkomst']}{datum}")
        if pd.notna(r["kans_boom"]):
            delen.append(f"beeld: kans boom {r['kans_boom']:.2f} ({r['verloop']})")
        if r.get("eigen_kroon_ahn5"):
            delen.append(f"eigen kroon in AHN5 ({r['ahn5_kroon_hoogte_m']:.0f} m)")
        if pd.notna(r["laatste_bron"]):
            delen.append(f"nieuwste bewijs: {r['laatste_bron']} {r['laatste_datum']:%Y-%m-%d} ({r['toestand']})")
        return "; ".join(delen)

    t["bewijs"] = t.apply(bewijs, axis=1)
    return t


# ------------------------------------------------------------------- toetsen


def toetsen(t: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Hoe goed kloppen de bronnen met elkaar? Een paar kruistabellen voor het rapport."""
    uit = {}
    uit["status_x_zekerheid"] = pd.crosstab(t["status"], t["zekerheid"], margins=True, margins_name="totaal").reset_index()
    uit["status_x_omgeving"] = pd.crosstab(t["status"], t["omgeving"], margins=True, margins_name="totaal").reset_index()

    # Ziet het beeld een herplante boom? Per aantal jaren sinds het planten.
    geplant = t[t["plantdatum"].notna() & t["kans_boom"].notna()].copy()
    if len(geplant):
        geplant["jaren_sinds_planten"] = ((pd.Timestamp("2026-04-01") - geplant["plantdatum"]).dt.days / 365.25).clip(lower=0).round()
        uit["beeld_bij_herplant"] = (geplant.groupby("jaren_sinds_planten")
                                     .agg(stobbes=("id", "size"), beeld_ziet_boom_pct=("kans_boom", lambda k: round(100 * (k >= 0.5).mean(), 1)))
                                     .reset_index())
    # Beeld tegen de inspecteur: onafhankelijk van elkaar vastgesteld
    insp = t[t["herplantinspectie_uitkomst"].isin(["Ja", "Nee", "Ja, nieuwe boom reeds aangeplant"]) & t["kans_boom"].notna()]
    if len(insp):
        uit["beeld_x_inspectie"] = (insp.groupby("herplantinspectie_uitkomst")
                                    .agg(stobbes=("id", "size"), beeld_ziet_boom_pct=("kans_boom", lambda k: round(100 * (k >= 0.5).mean(), 1)),
                                         mediane_kans=("kans_boom", "median"))
                                    .reset_index())
    uit["herplant_status"] = t["herplant_status"].value_counts().rename_axis("herplant_status").reset_index(name="stobbes")
    return uit


# ------------------------------------------------------------------- rapport


def _getal(waarde: float, decimalen: int = 0) -> str:
    """Nederlandse notatie: punt voor duizendtallen, komma voor decimalen."""
    return f"{waarde:,.{decimalen}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def schrijf_rapport(t: pd.DataFrame, toets: dict, beeld: dict, pad: Path) -> Path:
    n = len(t)
    telling = t["status"].value_counts()
    regels = [
        f"# Stobbes in Amsterdam: correct of al vervangen ({date.today():%d-%m-%Y})",
        "",
        f"Geanalyseerd: {_getal(n)} boompunten met type Stobbe uit de bomen-WFS (filter type_object = Stobbe).",
        "",
        "## Uitkomst",
        "",
        "| status | stobbes | aandeel | advies |",
        "|---|---|---|---|",
    ]
    for sleutel, tekst in STATUS.items():
        aantal = int(telling.get(tekst, 0))
        regels.append(f"| {tekst} | {_getal(aantal)} | {_getal(100 * aantal / n, 1)}% | {ADVIES[sleutel]} |")
    regels += [
        "",
        "## Hoe het bepaald is",
        "",
        "Per stobbe zijn vier soorten bewijs naast elkaar gelegd. De registratie zelf: staat er een levende boom "
        f"binnen {_getal(HERPLANT_OP_PLEK_M, 1)} m in de bomenkaart, dan is de herplant als nieuw punt vastgelegd en had de "
        "stobbe moeten verdwijnen. Het kap- en herplanttraject: kapdatum, plantdatum, acceptatie van het werk en de "
        "uitkomst van de herplantinspectie. Het beeld: hoogte en kroon uit het AHN en het vegetatiegetal van de "
        "infraroodluchtfoto's per jaar, gecombineerd in een model dat getoetst is op stobbes waarvan bekend is of de "
        "boom er tijdens de vlucht nog stond. En straatbeeld, om met een klik zelf te kijken.",
        "",
        f"Het beeldmodel koos de combinatie **{beeld['keuze']}**. Scores per combinatie (kruisvalidatie per kilometerblok):",
        "",
        tabel_md(beeld["combinaties"][beeld["combinaties"]["jaar"].astype(str) == "alle"]
                 [["combinatie", "stobbes_met_antwoord", "auc", "auc_binnen_jaar", "gebalanceerd_juist"]]),
        "",
        "## Wat de bronnen wel en niet kunnen",
        "",
        "- **Registratie en traject** zijn het sterkste bewijs, want het is wat de gemeente zelf heeft vastgelegd. "
        "Een stobbe met een levende boom op hetzelfde punt is vrijwel zeker een vergeten afvoer.",
        "- **Het AHN is oud**: AHN4 is van 2020-2022, AHN5 boven Amsterdam van rond voorjaar 2023. Wat daarna gekapt "
        "of geplant is, ziet het niet. AHN6 (2025) is voor Amsterdam nog niet vrijgegeven.",
        "- **Jonge bomen zijn op een luchtfoto lastig**: een kroon van een meter is op 21-25 cm een paar pixels. "
        "Zie de tabel hieronder: hoe vaak het beeld een herplante boom ziet, per jaar na het planten.",
        "- **Bomen op de kaplijst zijn vaak al dood**: zonder blad geen groen op het infrarood, terwijl ze er staan.",
        "- **In bos en dichte groepen** hangt de kroon van de buren boven de stobbe. Daar is straatbeeld nodig.",
        "",
    ]
    for naam, kop in (("beeld_bij_herplant", "Ziet het beeld een herplante boom?"),
                      ("beeld_x_inspectie", "Beeld tegen de herplantinspectie (onafhankelijk)"),
                      ("status_x_omgeving", "Status per omgeving"), ("herplant_status", "Herplantstatus volgens het traject")):
        if naam in toets:
            regels += [f"### {kop}", "", tabel_md(toets[naam]), ""]
    regels += [
        "## Nalopen",
        "",
        "Open `stobbes_analyse.gpkg` in QGIS. De laag `te_controleren` bevat alleen wat aandacht vraagt, op "
        "prioriteit. Per stobbe staan drie links: `panorama_beeld` (het nieuwste Amsterdamse panorama, gericht op de "
        "plek, met `panorama_datum`), `google_streetview` en `google_maps`. In QGIS: rechtsklik op de laag, "
        "Eigenschappen, Acties, of gebruik de identificatieknop en klik de link aan.",
        "",
        "## Nauwkeuriger bronnen voor een volgende stap",
        "",
        "- **De LiDAR-puntenwolk van Amsterdam (2025)**: gemeten vanaf de panoramawagen, dus stammen en jonge bomen "
        "langs de straat zijn erin te zien. De index staat open (api.data.amsterdam.nl/v1/puntenwolk), maar de "
        "bestandsserver files.lidar.data.amsterdam.nl was op het moment van deze analyse niet bereikbaar.",
        "- **AHN6** zodra Amsterdam is vrijgegeven: hoogte van 2025, dus na het grootste deel van de kap.",
        "- **Beeldherkenning op de panoramabeelden**: een detectiemodel dat boompalen en jonge stammen herkent op de "
        "zomerpanorama's van 2025 zou herplant per straat kunnen bevestigen zonder dat iemand hoeft te kijken.",
        "- **De afgeschermde velden van de API** (FP/MDW): kroondiameter en boomhoogte uit de veiligheidsinspecties "
        "zijn met een token op te vragen en geven per boom een recente veldmeting.",
    ]
    pad.write_text("\n".join(regels) + "\n", encoding="utf-8")
    return pad


# ------------------------------------------------------------------- uitvoeren


def analyseer_stobbes(bbox=None, opnieuw: bool = False, straatbeeld_ophalen: bool = True,
                      kopie_naar_onedrive: bool = True) -> gpd.GeoDataFrame:
    boompunten = bomenregister.alle_boompunten()
    stobbes, bomen = bomenregister.splits(boompunten)
    kap = bomenregister.kap_en_herplant()
    if bbox is not None:
        stobbes = stobbes.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True)
        logger.info("Proef op %s: %s stobbes", bbox, len(stobbes))
    stobbes["punt_id"] = puntmeting.punt_ids(stobbes, "stobbe")
    logger.info("%s stobbes, %s levende bomen", f"{len(stobbes):,}", f"{len(bomen):,}")

    beeld = stobbes_controle.bereken_beeldkansen(stobbes, kap, opnieuw)
    beeldkolommen = ["punt_id", "kans_boom", "boom_mogelijk_aanwezig", "verloop", "oordeel", "groen_aantal_jaargangen",
                     "kroon_ahn5", "geen_groen_geen_kroon"]
    beeldkolommen += [k for k in beeld["tabel"].columns if k.startswith(("kans_boom_", "boom_20", "groen_20", "ahn", "ndvi_"))]
    t = stobbes.merge(beeld["tabel"][list(dict.fromkeys(beeldkolommen))].rename(columns={"oordeel": "beeld_oordeel"}),
                      on="punt_id", how="left")
    t = t.merge(buurbomen(stobbes, bomen), on="punt_id", how="left")
    t = t.merge(traject(stobbes, kap), on="id", how="left")
    t = t.merge(kroon_van_stobbe(stobbes, bomen), on="punt_id", how="left")
    om = bomenregister.omgeving(stobbes, bomen)[["omgeving", "bomen_binnen_5m"]]
    om["punt_id"] = stobbes["punt_id"].values
    t = t.merge(om, on="punt_id", how="left")

    t = beoordeel(t)
    if straatbeeld_ophalen:
        links = straatbeeld.panorama_bij_punten(stobbes, "stobbes")
    else:
        links = straatbeeld.links(stobbes)
    t = t.merge(links.drop(columns=["lat", "lon"], errors="ignore"), on="punt_id", how="left")

    toets = toetsen(t)
    UITVOER_MAP.mkdir(parents=True, exist_ok=True)
    pad = UITVOER_MAP / "stobbes_analyse.gpkg"
    voorop = ["id", "status", "zekerheid", "advies", "prioriteit", "bewijs", "panorama_beeld", "panorama_datum",
              "google_streetview", "google_maps", "omgeving", "herplant_status", "toestand", "laatste_bron", "laatste_datum",
              "kans_boom", "verloop",
              "buurboom_id", "buurboom_afstand_m", "andere_stobbe_binnen_1m"]
    rest = [k for k in t.columns if k not in voorop + ["geometry", "punt_id", "status_sleutel"]]
    uitvoer = gpd.GeoDataFrame(t[voorop + rest + ["status_sleutel", "geometry"]], geometry="geometry", crs="EPSG:28992")
    uitvoer = uitvoer.sort_values("prioriteit")
    if pad.exists():
        pad.unlink()
    maak_schrijfbaar(uitvoer).to_file(pad, layer="stobbes", driver="GPKG", dataset_options={"VERSION": "1.2"})
    controle = uitvoer[uitvoer["status_sleutel"].isin(["fout_zelfde_plek", "fout_dubbel", "fout_niet_geregistreerd",
                                                       "mogelijk_boom", "onzeker"])]
    maak_schrijfbaar(controle).to_file(pad, layer="te_controleren", driver="GPKG")

    nieuw = bomen[bomen["id"].isin(t.loc[t["buurboom_afstand_m"].le(HERPLANT_DICHTBIJ_M), "buurboom_id"])]
    maak_schrijfbaar(nieuw).to_file(pad, layer="levende_boom_bij_stobbe", driver="GPKG")

    overzicht = (t.groupby(["status", "zekerheid"]).size().rename("stobbes").reset_index())
    tabellen = {"overzicht": overzicht, "beeld_combinaties": beeld["combinaties"], "beeld_per_jaar": beeld["per_jaar"],
                "ahn_peiljaar": beeld["ahn_peiljaar"], **toets}
    for naam, tabel in tabellen.items():
        pyogrio.write_dataframe(tabel, pad, layer=naam, driver="GPKG")
    rapport = schrijf_rapport(t, toets, beeld, UITVOER_MAP / "rapport_stobbes.md")

    try:
        from luchtfoto_objecten.qgis_stijl import zet_stijl
        zet_stijl(pad, ["stobbes", "te_controleren"], "status", {STATUS[k]: KLEUR[k] for k in STATUS})
    except Exception as fout:  # de opmaak is een extraatje, de analyse staat er ook zonder
        logger.warning("QGIS-opmaak niet toegevoegd: %s", fout)

    print("\n" + rapport.read_text(encoding="utf-8").split("## Hoe het bepaald is")[0])
    print(f"GeoPackage: {pad}\nRapport:    {rapport}")
    opslag.werk_leesmij_bij()
    if kopie_naar_onedrive and bbox is None and ONEDRIVE_BOMEN.exists():
        stempel = f"{date.today():%d%m%Y}"
        shutil.copyfile(pad, ONEDRIVE_BOMEN / f"Stobbes_analyse_{stempel}.gpkg")
        shutil.copyfile(rapport, ONEDRIVE_BOMEN / f"Stobbes_analyse_{stempel}.md")
        print(f"Kopie in de datamap: {ONEDRIVE_BOMEN}")
    return uitvoer
