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


@dataclass
class GroenParameters:
    exg_ondergrens: float = 0.035
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
