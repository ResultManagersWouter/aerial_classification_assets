from __future__ import annotations

from dataclasses import dataclass

GROEN = "groen"
VERHARDING = "verharding"


@dataclass(frozen=True)
class Thema:
    naam: str
    bronnen: tuple[str, ...]
    detectieklasse: str
    registratieset: str
    omschrijving: str


# De beheerregistratie beschrijft wat de gemeente onderhoudt, de BGT beschrijft de
# topografie. Beide zouden de openbare ruimte vlakdekkend moeten beslaan, en juist het
# verschil met de luchtfoto laat zien waar de administratie achterloopt.
#
# Het verharde maaiveld staat in de BGT verdeeld over wegdelen en onbegroeide
# terreindelen. Die horen in één thema, anders meldt elke rijbaan zich als ontbrekend
# onbegroeid terreindeel.
THEMAS: list[Thema] = [
    Thema(
        naam="groen_beheer",
        bronnen=("groenobjecten",),
        detectieklasse=GROEN,
        registratieset="beheerregistratie",
        omschrijving="Beheerde groenvlakken tegen de vegetatie op de luchtfoto",
    ),
    Thema(
        naam="verharding_beheer",
        bronnen=("verhardingen",),
        detectieklasse=VERHARDING,
        registratieset="beheerregistratie",
        omschrijving="Beheerde verhardingsvlakken tegen het onbegroeide maaiveld",
    ),
    Thema(
        naam="groen_bgt",
        bronnen=("begroeideterreindelen",),
        detectieklasse=GROEN,
        registratieset="bgt",
        omschrijving="Begroeide terreindelen uit de BGT tegen de vegetatie op de luchtfoto",
    ),
    Thema(
        naam="verharding_bgt",
        bronnen=("wegdelen", "onbegroeideterreindelen"),
        detectieklasse=VERHARDING,
        registratieset="bgt",
        omschrijving="Wegdelen en onbegroeide terreindelen samen tegen het onbegroeide maaiveld",
    ),
]

MASKERBRONNEN = ["panden", "waterdelen", "beheerkaart"]
REGISTRATIESETS = ["beheerregistratie", "bgt"]


def bronnen_voor_analyse() -> list[str]:
    bronnen = [bron for thema in THEMAS for bron in thema.bronnen]
    return list(dict.fromkeys(bronnen)) + MASKERBRONNEN


def themas_van_set(registratieset: str) -> list[Thema]:
    return [thema for thema in THEMAS if thema.registratieset == registratieset]
