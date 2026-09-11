from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

PROJECT_MAP = Path(__file__).resolve().parent.parent
PARAMETERS_BESTAND = PROJECT_MAP / "config" / "parameters.yaml"
DATA_MAP = PROJECT_MAP / "data"


@dataclass
class LuchtfotoParameters:
    laag: str = "Actueel_orthoHR"
    zoom: int = 15
    max_werkers: int = 4
    groenlaag: str = "auto"
    groen_minimaal_aandeel: float = 0.10
    # Waarop de classificatie rust. Infrarood staat voor: NDVI scheidt vegetatie van
    # verharding veel scherper dan een kleurindex, gemeten 84% van het geregistreerde
    # groen bij 7% vals op wegdelen tegen enkele procenten precisie op kleur.
    beeld: str = "infrarood"


@dataclass
class GroenParameters:
    exg_ondergrens: float = 0.035
    # Ondergrens voor NDVI op de infraroodopname. Visueel beoordeeld op de beelden: onder
    # 0,05 loopt er te veel verharding mee. De ijking op de registratie mag wel strenger
    # uitkomen, maar niet losser; die kiest soms een negatieve drempel omdat de registratie
    # zelf ruimer is dan wat er op de foto staat.
    ndvi_ondergrens: float = 0.05
    min_oppervlakte_m2: float = 3.0
    vereenvoudiging_m: float = 0.20
    textuur_venster_m: float = 1.0


@dataclass
class VerhardingParameters:
    max_verzadiging: float = 0.35
    min_helderheid: float = 0.18
    max_helderheid: float = 0.95
    max_textuur: float = 0.10
    min_oppervlakte_m2: float = 8.0
    vereenvoudiging_m: float = 0.25
    gebouwbuffer_m: float = 0.5


@dataclass
class GroenstructuurParameters:
    """Scheidt boomkronen van groen op het maaiveld.

    De zomeropname laat zien waar blad hangt, de 8cm-voorjaarsopname is met kale bomen
    gevlogen en laat dus de grond onder die kroon zien. Wat daar niet verhard is, is
    vrijwel altijd gras of beplanting, ook als er een boom boven staat.

    Het groensignaal is leidend, de onverharde ondergrond telt mee als steun en ruwe
    textuur pleit juist voor een kroon. Met de gewichten kun je die afweging verschuiven.
    """

    gewicht_groen: float = 1.0
    gewicht_onverhard: float = 0.5     # steun, niet leidend
    gewicht_ruw: float = 0.75          # ruwe textuur pleit tegen een groenstrook
    drempel: float = 0.7               # hierboven telt een pixel als groenstrook
    textuurdrempel: float = 0.0        # 0 = bepaal met Otsu binnen het groen
    kroon_min_oppervlakte_m2: float = 2.0
    strook_min_oppervlakte_m2: float = 5.0
    vereenvoudiging_m: float = 0.25

    # Een boom herken je aan de vorm van zijn kroon: rond, compact en niet breed. Een
    # grasveld of berm is juist uitgestrekt of langgerekt. Deze drie grenzen scheiden
    # de twee, en een vlak moet ze alle drie halen om als boom te tellen.
    boom_max_breedte_m: float = 9.0        # hydraulische breedte, ruwweg de kroondiameter
    boom_min_compactheid: float = 0.30     # 1,0 is een perfecte cirkel
    boom_max_oppervlakte_m2: float = 250.0


@dataclass
class SignaleringParameters:
    min_dekkingsgraad: float = 0.55
    lage_dekkingsgraad: float = 0.20
    min_nieuw_oppervlakte_m2: float = 15.0
    min_registratie_m2: float = 2.0
    min_witte_vlek_m2: float = 25.0
    sliverbreedte_m: float = 0.75
    celgrootte_m: float = 100.0
    cel_signaaldrempel: int = 5
    beperk_tot_beheergebied: bool = True


@dataclass
class Instellingen:
    luchtfoto: LuchtfotoParameters = field(default_factory=LuchtfotoParameters)
    groen: GroenParameters = field(default_factory=GroenParameters)
    verharding: VerhardingParameters = field(default_factory=VerhardingParameters)
    groenstructuur: GroenstructuurParameters = field(default_factory=GroenstructuurParameters)
    signalering: SignaleringParameters = field(default_factory=SignaleringParameters)
    data_map: Path = DATA_MAP

    @classmethod
    def laden(cls, bestand: Path | None = None) -> "Instellingen":
        bestand = bestand or PARAMETERS_BESTAND
        if not bestand.exists():
            return cls()
        ruwe_data = yaml.safe_load(bestand.read_text(encoding="utf-8")) or {}
        instellingen = cls()
        for veld in fields(cls):
            blok = ruwe_data.get(veld.name)
            if not isinstance(blok, dict):
                continue
            groep = getattr(instellingen, veld.name)
            bekend = {subveld.name for subveld in fields(groep)}
            onbekend = set(blok) - bekend
            if onbekend:
                raise ValueError(f"Onbekende parameters in {veld.name}: {sorted(onbekend)}")
            for sleutel, waarde in blok.items():
                setattr(groep, sleutel, waarde)
        return instellingen

    @property
    def cache_map(self) -> Path:
        return self.data_map / "cache"

    @property
    def uitvoer_map(self) -> Path:
        return self.data_map / "uitvoer"
