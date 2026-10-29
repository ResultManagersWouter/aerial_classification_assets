"""Stobbes controleren op beeld: stond er op de plek van een stobbe in 2024, 2025 of 2026 een boom?

Dit is het beelddeel van de stobbe-analyse: een kans per stobbe per jaar, uit open
hoogte- en luchtfotodata. De volledige analyse, met de registratie, het kap- en
herplanttraject en straatbeeld erbij, staat in stobbes_analyse.py en draait via
stobbe_main.py. Dit script kan ook los, voor alleen de beeldkansen.

Invoer per stobbe:

    vegetatiegetal   NDVI uit de infraroodopnamen van PDOK, per jaargang
    objecthoogte     AHN4 (2020-2022) en AHN5 (Amsterdam rond voorjaar 2023), met daaruit
                     hoogte bij de stam, hoogteruwheid en kroonvolume binnen 3 m

Welke bronnen het oordeel dragen wordt gemeten, niet vooraf gekozen. De toets zit in de
WFS zelf: de laag kapenherplant geeft per boom de kapdatum en de herplantinspecties. Is
een boom ná de vlucht van een opname gekapt, of zag een inspecteur hem later nog staan,
dan stond hij er tijdens de opname. Is hij ervóór gekapt, of zag een inspecteur er eerder
al geen boom, dan niet. Zo ontstaat een set stobbes met een bekend antwoord, op precies de
plekken waar het om gaat: in bomenrijen, in parken, op stoepen.

Op die set worden vijf combinaties gescoord met kruisvalidatie, waarbij de stad in blokken
van een kilometer wordt verdeeld zodat buren niet elkaars antwoord verraden:

    AHN                             alleen hoogte, AHN4 en AHN5
    satelliet, dit jaar             alleen het vegetatiegetal van die jaargang
    satelliet, alle jaargangen      het vegetatiegetal van 2023 tot en met 2026 samen
    AHN + satelliet, dit jaar
    AHN + satelliet, alle jaargangen

Het is één model over de drie jaren samen, met een rij per stobbe per jaar. Het AHN is in
elk jaar hetzelfde, dus een verschil tussen jaren kan alleen uit de beelden komen. De
combinatie met de hoogste gemiddelde AUC binnen de jaren wint; scheelt het minder dan
0,01, dan de eenvoudigste, en satelliet alleen telt als eenvoudiger dan met AHN erbij.

Twee dingen die de proef in Noord liet zien. Het AHN is van voor de meeste kap (AHN5 is
boven Amsterdam rond voorjaar 2023 gevlogen) en ziet bij een stobbe in een bomenrij ook
de kroon van de buren. En bomen die op de kaplijst staan zijn vaak al dood: zonder blad
geven ze op het infrarood geen groen, terwijl ze er wel staan.

Draaien:

    python stobbes_controle.py                                      # alle stobbes
    python stobbes_controle.py --bbox 123000 489000 124500 490500   # proef op een stuk stad
    python stobbes_controle.py --opnieuw                            # kenmerken opnieuw meten

Alle downloads (WFS, AHN, luchtfototegels) en metingen per punt worden bewaard in de
gedeelde map "data assets Amsterdam", zie luchtfoto_objecten/opslag.py.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from luchtfoto_objecten import bomenregister, opslag, puntmeting
from luchtfoto_objecten.geo_hulp import maak_schrijfbaar
from luchtfoto_objecten.puntmeting import AHN_KENMERKEN, KROONHOOGTE_M, NDVI_KENMERKEN

logger = logging.getLogger("stobbes")

AHN_VERSIES = (4, 5)

# Infraroodopnamen. Per jaar het doel waarop het oordeel valt, plus 2023 en het voorjaar van
# 2025 als extra jaargangen voor het verloop. Gemeten op 120 straatbomen tegen 120 plekken
# zonder boom scheidt 2024_ortho25IR met AUC 0,95, 2025_ortho25IR 0,91 en 2026_orthoHRIR 0,90.
DOELLAGEN = {2024: "2024_ortho25IR", 2025: "2025_ortho25IR", 2026: "2026_orthoHRIR"}
LAGEN = ["2023_ortho25IR", "2024_ortho25IR", "2025_orthoHRIR", "2025_ortho25IR", "2026_orthoHRIR"]

# Geschatte vliegperiode per opname, ruim genomen. De 25cm-opname wordt in de zomer gevlogen,
# de 8cm-opname in het vroege voorjaar. Een gebeurtenis binnen de periode zegt niets over de
# opname en telt niet mee in de toets.
VLUCHT = {
    "2023_ortho25IR": ("2023-05-01", "2023-09-30"),
    "2024_ortho25IR": ("2024-05-01", "2024-09-30"),
    "2025_ortho25IR": ("2025-05-01", "2025-09-30"),
    "2025_orthoHRIR": ("2025-01-01", "2025-05-15"),
    "2026_orthoHRIR": ("2026-01-01", "2026-05-15"),
}

UITVOER_MAP = Path("output/stobbes")
ONEDRIVE_DATA = Path(
    "/Users/wks/Library/CloudStorage/OneDrive-SharedLibraries-Result!Managers/"
    "Amsterdam - datakwaliteitsdashboard/bomen/data"
)
GRENS_TWIJFEL = (0.35, 0.65)
AUC_MARGE = 0.01


def ahn_kolommen(versies=AHN_VERSIES) -> list[str]:
    return [f"ahn{v}_{k}" for v in versies for k in AHN_KENMERKEN]


def ndvi_kolommen(lagen) -> list[str]:
    return [f"{k}_{laag}" for laag in lagen for k in NDVI_KENMERKEN]


# ------------------------------------------------------------------ bekend antwoord


def bekende_antwoorden(stobbes: pd.DataFrame, kap: pd.DataFrame) -> pd.DataFrame:
    """Per stobbe en per opname: stond er toen een boom (1), niet (0), of weten we het niet.

    Wel een boom: de kap was na de vlucht, of een inspecteur zag na de vlucht nog een boom.
    Geen boom: de kap was voor de vlucht, of een inspecteur zag voor de vlucht al geen boom.
    Stobbes waar later een nieuwe boom is aangeplant laten we buiten de 'geen boom'-kant,
    want dan kan er op de opname een jonge boom staan. Spreken de signalen elkaar tegen,
    dan telt de stobbe voor die opname niet mee.
    """
    bij = kap[kap["boom_id"].isin(stobbes["id"])]
    kapdatum = bij.groupby("boom_id")["kapmaatregel_datum_uitgevoerd"].max()
    inspecties = bij.dropna(subset=["datum_herplantinspectie"])
    herplant = set(inspecties.loc[inspecties["boom_aanwezigheid"].eq("Ja, nieuwe boom reeds aangeplant"), "boom_id"])

    uit = pd.DataFrame({"id": stobbes["id"].values})
    k = uit["id"].map(kapdatum)
    for laag, (begin, eind) in VLUCHT.items():
        begin, eind = pd.Timestamp(begin), pd.Timestamp(eind)
        nee_voor = inspecties[(inspecties["boom_aanwezigheid"] == "Nee") & (inspecties["datum_herplantinspectie"] < begin)]
        ja_na = inspecties[(inspecties["boom_aanwezigheid"] == "Ja") & (inspecties["datum_herplantinspectie"] > eind)]
        boom = k.gt(eind) | uit["id"].isin(ja_na["boom_id"])
        geen = (k.lt(begin) | uit["id"].isin(nee_voor["boom_id"])) & ~uit["id"].isin(herplant)
        uit[f"antwoord_{laag}"] = pd.Series(np.where(boom & ~geen, 1.0, np.where(geen & ~boom, 0.0, np.nan)))
    uit["kapdatum"] = k
    return uit


# ------------------------------------------------------------------------ meten


def kenmerken(stobbes: gpd.GeoDataFrame, opnieuw: bool = False) -> pd.DataFrame:
    """AHN- en NDVI-kenmerken per stobbe, bewaard in de assetmap onder afgeleid/stobbes/."""
    tabel = stobbes[["punt_id"]].copy()
    for versie in AHN_VERSIES:
        gemeten = puntmeting.bewaard("stobbes", f"ahn{versie}", stobbes,
                                     lambda p, v=versie: puntmeting.meet_ahn(p, v), opnieuw)
        tabel = tabel.merge(gemeten, on="punt_id", how="left")
    for laag in LAGEN:
        gemeten = puntmeting.bewaard("stobbes", laag, stobbes,
                                     lambda p, laag=laag: puntmeting.meet_vegetatiegetal(p, laag), opnieuw)
        tabel = tabel.merge(gemeten.rename(columns={k: f"{k}_{laag}" for k in NDVI_KENMERKEN}), on="punt_id", how="left")
    return tabel


# ------------------------------------------------------------- combinaties scoren
#
# Eén model over alle jaren samen, met een rij per stobbe per jaar. Het AHN is voor elk jaar
# van een stobbe hetzelfde; wat per jaar verschilt is het vegetatiegetal van de opname van
# dat jaar. Zo kan een verschil tussen jaren alleen uit de beelden komen. Met een los model
# per jaar zou de kans ook verschuiven doordat de toetssets per jaar anders zijn
# samengesteld, en dat is geen verandering op straat.

# Bewust geen kenmerk voor het jaar of het seizoen: 2026 is de enige voorjaarsopname, en dan
# leert het model het gemiddelde van dat jaar in plaats van wat er op de plek staat.
JAAR_NDVI = [f"jaar_{k}" for k in NDVI_KENMERKEN]

COMBINATIES = {
    "AHN": ahn_kolommen(),
    "satelliet, dit jaar": JAAR_NDVI,
    "satelliet, alle jaargangen": JAAR_NDVI + ndvi_kolommen(LAGEN),
    "AHN + satelliet, dit jaar": ahn_kolommen() + JAAR_NDVI,
    "AHN + satelliet, alle jaargangen": ahn_kolommen() + JAAR_NDVI + ndvi_kolommen(LAGEN),
}
EENVOUD = {"satelliet, dit jaar": 0, "satelliet, alle jaargangen": 1, "AHN": 2,
           "AHN + satelliet, dit jaar": 3, "AHN + satelliet, alle jaargangen": 4}


def per_stobbe_per_jaar(tabel: pd.DataFrame) -> pd.DataFrame:
    """Lange tabel: een rij per stobbe per jaar, met het vegetatiegetal van dat jaar.

    Het vegetatiegetal wordt per opname rechtgezet op de mediaan van alle stobbes, want
    elke jaargang heeft zijn eigen kleurzweem (zie de README, 'Meer jaargangen').
    """
    delen = []
    for jaar, laag in DOELLAGEN.items():
        deel = tabel[["punt_id", "x", "y"] + ahn_kolommen() + ndvi_kolommen(LAGEN)].copy()
        deel["jaar"] = jaar
        for k in NDVI_KENMERKEN:
            waarde = tabel[f"{k}_{laag}"]
            deel[f"jaar_{k}"] = waarde - waarde.median() if k != "ndvi_contrast" else waarde
        deel["antwoord"] = tabel[f"antwoord_{laag}"]
        delen.append(deel)
    return pd.concat(delen, ignore_index=True)


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Beslisbomen, omdat een boom pas een boom is als signalen samengaan (groen bij de stam
    # en groener dan de omgeving), en omdat ze met ontbrekende metingen overweg kunnen.
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=300, min_samples_leaf=30,
                                          class_weight="balanced", random_state=1)


def scoor_combinaties(lang: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Kruisgevalideerde score per combinatie, totaal en per jaar, op stobbes met een bekend antwoord.

    De stad wordt in blokken van een kilometer verdeeld en elk blok zit in precies één
    vouw, met alle jaren van een stobbe erin. Zo zitten buren in een bomenrij en dezelfde
    stobbe in een ander jaar nooit tegelijk in training en toets.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    bekend = lang[lang["antwoord"].notna() & lang["jaar_ndvi_kern"].notna()]
    y = bekend["antwoord"].astype(int).values
    blokken = (bekend["x"] // 1000).astype(int).astype(str) + "_" + (bekend["y"] // 1000).astype(int).astype(str)
    if len(np.unique(y)) < 2 or blokken.nunique() < 2:
        raise SystemExit("Te weinig stobbes met een bekend antwoord om de combinaties te scoren")
    vouwen = GroupKFold(min(5, blokken.nunique()))

    def score(kans, waar, naam, jaar):
        return {"combinatie": naam, "jaar": jaar, "stobbes_met_antwoord": len(waar), "waarvan_boom": int(waar.sum()),
                "auc": round(roc_auc_score(waar, kans), 3) if len(np.unique(waar)) == 2 else None,
                "gebalanceerd_juist": round(balanced_accuracy_score(waar, kans >= 0.5), 3)}

    # Gekozen wordt op de gemiddelde AUC binnen de jaren. De AUC over alle jaren samen beloont
    # ook het verschil tussen jaren, en dat zegt meer over welke stobbes in welk jaar een
    # bekend antwoord hebben dan over wat er op straat staat.
    rijen, binnen_jaar = [], {}
    for naam, kolommen in COMBINATIES.items():
        kans = cross_val_predict(_model(), bekend[kolommen], y, groups=blokken, cv=vouwen, method="predict_proba")[:, 1]
        rijen.append(score(kans, y, naam, "alle"))
        per_jaar = []
        for jaar in DOELLAGEN:
            in_jaar = (bekend["jaar"] == jaar).values
            rijen.append(score(kans[in_jaar], y[in_jaar], naam, jaar))
            if rijen[-1]["auc"] is not None:
                per_jaar.append(rijen[-1]["auc"])
        binnen_jaar[naam] = float(np.mean(per_jaar))
        rijen[-len(DOELLAGEN) - 1]["auc_binnen_jaar"] = round(binnen_jaar[naam], 3)
    # Twee losse kenmerken als ijkpunt, zonder model: alleen groen bij de stam, of alleen
    # het kroonvolume uit AHN5
    for naam, kolom in (("ijkpunt: groen bij de stam", "jaar_ndvi_kern"), ("ijkpunt: kroonvolume AHN5", "ahn5_kroonvolume_m3")):
        gevuld = bekend[kolom].notna().values
        rijen.append({"combinatie": naam, "jaar": "alle", "stobbes_met_antwoord": int(gevuld.sum()),
                      "waarvan_boom": int(y[gevuld].sum()),
                      "auc": round(roc_auc_score(y[gevuld], bekend.loc[gevuld, kolom]), 3)})

    beste = max(binnen_jaar.values())
    keuze = min((n for n, s in binnen_jaar.items() if beste - s <= AUC_MARGE), key=EENVOUD.get)
    scores = pd.DataFrame(rijen)
    scores["gekozen"] = scores["combinatie"].eq(keuze)
    return scores, keuze


def voorspel(lang: pd.DataFrame, keuze: str) -> pd.DataFrame:
    """Kans op een boom per stobbe per jaar, met de gekozen combinatie getraind op alle bekende antwoorden."""
    kolommen = COMBINATIES[keuze]
    bekend = lang[lang["antwoord"].notna() & lang["jaar_ndvi_kern"].notna()]
    model = _model().fit(bekend[kolommen], bekend["antwoord"].astype(int))
    kans = pd.Series(model.predict_proba(lang[kolommen])[:, 1], index=lang.index)
    # Zonder beeld van dat jaar geen oordeel over dat jaar
    kans[lang["jaar_ndvi_kern"].isna()] = np.nan
    breed = lang.assign(kans=kans.round(3)).pivot(index="punt_id", columns="jaar", values="kans")
    breed.columns = [f"kans_boom_{j}" for j in breed.columns]
    return breed.reset_index()


def groendrempels(tabel: pd.DataFrame) -> dict[str, float]:
    """Per opname de NDVI-drempel die boom en geen boom het best scheidt (Youden), voor de groendetectie."""
    from sklearn.metrics import roc_curve

    drempels = {}
    for laag in LAGEN:
        sel = tabel[tabel[f"antwoord_{laag}"].notna() & tabel[f"ndvi_kern_{laag}"].notna()]
        if sel[f"antwoord_{laag}"].nunique() < 2:
            continue
        fpr, tpr, grenzen = roc_curve(sel[f"antwoord_{laag}"], sel[f"ndvi_kern_{laag}"])
        drempels[laag] = float(grenzen[np.argmax(tpr - fpr)])
    return drempels


# ------------------------------------------------------------------------ oordeel


def oordeel(tabel: pd.DataFrame, drempels: dict[str, float]) -> pd.DataFrame:
    """Booleans per jaar, de groendetectie per jaargang, het verloop en een leesbaar oordeel."""
    uit = tabel.copy()
    jaren = [j for j in DOELLAGEN if f"kans_boom_{j}" in uit]
    for jaar in jaren:
        kans = uit[f"kans_boom_{jaar}"]
        uit[f"boom_{jaar}"] = kans.ge(0.5).astype("boolean").mask(kans.isna())

    # De groendetectie zoals je hem in QGIS zou doen: groen bij de stam, per jaargang
    for laag, drempel in drempels.items():
        ndvi = uit[f"ndvi_kern_{laag}"]
        uit[f"groen_{laag}"] = ndvi.gt(drempel).astype("boolean").mask(ndvi.isna())
    groen = uit[[f"groen_{laag}" for laag in drempels]]
    uit["groen_aantal_jaargangen"] = groen.fillna(False).sum(axis=1).astype(int)
    uit["kroon_ahn5"] = uit["ahn5_hoogte_max"].gt(KROONHOOGTE_M) & uit["ahn5_kroonvolume_m3"].gt(0)
    uit["geen_groen_geen_kroon"] = groen.notna().any(axis=1) & ~groen.fillna(False).any(axis=1) & ~uit["kroon_ahn5"]

    laatste = uit[f"kans_boom_{jaren[-1]}"]
    for jaar in reversed(jaren[:-1]):
        laatste = laatste.fillna(uit[f"kans_boom_{jaar}"])
    uit["kans_boom"] = laatste
    uit["boom_mogelijk_aanwezig"] = laatste.ge(0.5)

    def verloop(rij) -> str:
        reeks = {j: bool(rij[f"boom_{j}"]) for j in jaren if pd.notna(rij[f"boom_{j}"])}
        if not reeks:
            return "geen meting"
        if all(reeks.values()):
            return "boom in alle jaren"
        if not any(reeks.values()):
            return "geen boom in alle jaren"
        if not list(reeks.values())[-1]:
            return f"boom verdwenen, voor het eerst weg in {next(j for j, b in reeks.items() if not b)}"
        return f"boom verschenen na {[j for j, b in reeks.items() if not b][-1]}"

    uit["verloop"] = uit.apply(verloop, axis=1)
    uit["oordeel"] = np.select(
        [laatste.isna(),
         ~uit["boom_mogelijk_aanwezig"] & uit["geen_groen_geen_kroon"],
         laatste.between(*GRENS_TWIJFEL),
         uit["boom_mogelijk_aanwezig"],
         ~uit["boom_mogelijk_aanwezig"]],
        ["geen meting",
         "stobbe bevestigd: in geen jaargang groen en geen kroonvolume",
         "twijfel: beoordeel op de luchtfoto",
         "stobbe klopt niet: er staat een boom",
         "stobbe klopt: geen boom"],
        default="geen meting",
    )
    return uit


def ahn_peiljaar(tabel: pd.DataFrame) -> pd.DataFrame:
    """Wanneer is het AHN gevlogen? Na de kap hoort de hoogte bij de stam weg te zijn."""
    met_kap = tabel[tabel["kapdatum"].notna()]
    agg = {"stobbes": ("id", "size")}
    for v in AHN_VERSIES:
        agg[f"ahn{v}_pct_kroonhoogte"] = (f"ahn{v}_hoogte_max", lambda h: round(100 * (h > KROONHOOGTE_M).mean(), 1))
    return met_kap.groupby(met_kap["kapdatum"].dt.to_period("Q").astype(str).rename("kapkwartaal")).agg(**agg).reset_index()


def samenvatting_per_jaar(tabel: pd.DataFrame, keuze: str) -> pd.DataFrame:
    rijen = []
    for jaar in DOELLAGEN:
        kans = tabel[f"kans_boom_{jaar}"]
        rijen.append({
            "jaar": jaar, "opname": DOELLAGEN[jaar], "combinatie": keuze, "gemeten": int(kans.notna().sum()),
            "boom": int((kans >= 0.5).sum()), "geen_boom": int((kans < 0.5).sum()),
            "twijfel": int(kans.between(*GRENS_TWIJFEL).sum()),
            "pct_boom": round(100 * (kans >= 0.5).sum() / max(kans.notna().sum(), 1), 1),
        })
    return pd.DataFrame(rijen)


# ---------------------------------------------------------------------- uitvoeren


def bereken_beeldkansen(stobbes: gpd.GeoDataFrame, kap: pd.DataFrame, opnieuw: bool = False) -> dict:
    """Het hele beelddeel: meten, combinaties scoren, kansen per jaar en het oordeel.

    `stobbes` moet de kolommen id, punt_id en geometry hebben. Geeft een dict met de tabel
    per stobbe en de tabellen voor het rapport.
    """
    stobbes = stobbes.copy()
    stobbes["x"], stobbes["y"] = stobbes.geometry.x, stobbes.geometry.y
    tabel = (stobbes.merge(kenmerken(stobbes[["id", "punt_id", "geometry"]], opnieuw=opnieuw), on="punt_id", how="left")
             .merge(bekende_antwoorden(stobbes, kap), on="id", how="left"))
    lang = per_stobbe_per_jaar(tabel)
    scores, keuze = scoor_combinaties(lang)
    tabel = tabel.merge(voorspel(lang, keuze), on="punt_id", how="left")
    drempels = groendrempels(tabel)
    tabel = oordeel(tabel, drempels)
    return {
        "tabel": tabel, "combinaties": scores, "keuze": keuze,
        "per_jaar": samenvatting_per_jaar(tabel, keuze), "ahn_peiljaar": ahn_peiljaar(tabel),
        "groendrempels": pd.DataFrame({"opname": list(drempels), "ndvi_drempel_groen": [round(d, 4) for d in drempels.values()]}),
    }


def controleer_stobbes(bbox: tuple[float, float, float, float] | None = None, opnieuw: bool = False,
                       kopie_naar_onedrive: bool = True) -> gpd.GeoDataFrame:
    """Alleen het beelddeel, met eigen uitvoer in output/stobbes/."""
    stobbes, _ = bomenregister.splits(bomenregister.alle_boompunten())
    kap = bomenregister.kap_en_herplant()
    if bbox is not None:
        stobbes = stobbes.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True)
        logger.info("Proef op %s: %s stobbes", bbox, len(stobbes))
    stobbes["punt_id"] = puntmeting.punt_ids(stobbes, "stobbe")
    uitkomst = bereken_beeldkansen(stobbes, kap, opnieuw)
    tabel = uitkomst["tabel"]
    oordelen = tabel["oordeel"].value_counts().rename_axis("oordeel").reset_index(name="stobbes")
    verloop = tabel["verloop"].value_counts().rename_axis("verloop").reset_index(name="stobbes")

    pd.set_option("display.width", 200)
    print("\nCombinaties gescoord op stobbes met een bekend antwoord (kruisvalidatie per km-blok):")
    print(uitkomst["combinaties"].to_string(index=False))
    print(f"\nGekozen: {uitkomst['keuze']}")
    print("\nAHN bij stobbes per kapkwartaal, % met kroonhoogte bij de stam:")
    print(uitkomst["ahn_peiljaar"].to_string(index=False))
    print("\nStobbes per jaar:")
    print(uitkomst["per_jaar"].to_string(index=False))
    print("\nVerloop over de jaren:")
    print(verloop.to_string(index=False))
    print("\nOordeel:")
    print(oordelen.to_string(index=False))

    UITVOER_MAP.mkdir(parents=True, exist_ok=True)
    pad = UITVOER_MAP / "stobbes.gpkg"
    voorop = ["id", "boom_mogelijk_aanwezig", "oordeel", "kans_boom", "verloop"]
    voorop += [f"{k}_{j}" for j in DOELLAGEN for k in ("kans_boom", "boom")]
    voorop += [k for k in tabel.columns if k.startswith("groen_")] + ["kroon_ahn5", "geen_groen_geen_kroon"]
    wfs = [k for k in stobbes.columns if k not in ("punt_id", "geometry", "id")]
    meetwaarden = [k for k in tabel.columns if k not in voorop + wfs + ["punt_id", "x", "y", "geometry"]]
    uitvoer = gpd.GeoDataFrame(tabel[voorop + wfs + meetwaarden + ["geometry"]], geometry="geometry", crs="EPSG:28992")
    if pad.exists():
        pad.unlink()
    maak_schrijfbaar(uitvoer).to_file(pad, layer="stobbes", driver="GPKG")
    for naam in ("combinaties", "per_jaar", "groendrempels", "ahn_peiljaar"):
        pyogrio.write_dataframe(uitkomst[naam], pad, layer=naam, driver="GPKG")
        uitkomst[naam].to_csv(UITVOER_MAP / f"stobbes_{naam}.csv", index=False)
    for naam, tabel_uit in {"verloop": verloop, "oordeel": oordelen}.items():
        pyogrio.write_dataframe(tabel_uit, pad, layer=naam, driver="GPKG")
        tabel_uit.to_csv(UITVOER_MAP / f"stobbes_{naam}.csv", index=False)
    print(f"\nGeoPackage: {pad}")
    opslag.werk_leesmij_bij()

    if kopie_naar_onedrive and bbox is None and ONEDRIVE_DATA.exists():
        # Kopiëren buiten GDAL om: GDAL leest het uitroepteken in het OneDrive-pad als zip-scheiding
        doel = ONEDRIVE_DATA / f"Stobbes_boom_of_stobbe_{date.today():%d%m%Y}.gpkg"
        shutil.copyfile(pad, doel)
        print(f"Kopie in de datamap: {doel}")
    return uitvoer


def main(argumentenlijst: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Staat er op de plek van een stobbe toch nog een boom?")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="alleen de stobbes in dit gebied, in RD; handig als proef")
    parser.add_argument("--opnieuw", action="store_true", help="meet de kenmerken opnieuw in plaats van uit de opslag")
    parser.add_argument("--geen-kopie", action="store_true", help="niet naar de OneDrive-datamap kopiëren")
    argumenten = parser.parse_args(argumentenlijst)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    # Een AHN-tegel die niet bestaat (alleen water, buiten de dekking) geeft een 403; dat is
    # verwacht en wordt al afgevangen, dus die meldingen van GDAL hoeven niet in beeld
    logging.getLogger("rasterio._env").setLevel(logging.WARNING)
    controleer_stobbes(bbox=tuple(argumenten.bbox) if argumenten.bbox else None, opnieuw=argumenten.opnieuw,
                       kopie_naar_onedrive=not argumenten.geen_kopie)


if __name__ == "__main__":
    main()
