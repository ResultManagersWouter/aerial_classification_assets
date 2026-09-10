from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from shapely.geometry import Polygon, box

RD = "EPSG:28992"
PROJECT_MAP = Path(__file__).resolve().parent.parent
GEBIEDEN_BESTAND = PROJECT_MAP / "config" / "gebieden.yaml"


@dataclass(frozen=True)
class Gebied:
    naam: str
    bbox: tuple[float, float, float, float]
    omschrijving: str = ""
    stadsdeel: str = ""

    @property
    def breedte_m(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def hoogte_m(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def oppervlakte_ha(self) -> float:
        return self.breedte_m * self.hoogte_m / 10_000

    def als_polygon(self) -> Polygon:
        return box(*self.bbox)

    def __str__(self) -> str:
        return f"{self.naam} ({self.breedte_m:.0f} x {self.hoogte_m:.0f} m, {self.oppervlakte_ha:.1f} ha)"


def laad_gebieden(bestand: Path | None = None) -> dict[str, Gebied]:
    bestand = bestand or GEBIEDEN_BESTAND
    ruwe_data = yaml.safe_load(bestand.read_text(encoding="utf-8")) or {}
    gebieden = {}
    for naam, waarden in ruwe_data.items():
        bbox = tuple(float(getal) for getal in waarden["bbox"])
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"Gebied {naam} heeft een ongeldige bbox: {bbox}")
        gebieden[naam] = Gebied(
            naam=naam,
            bbox=bbox,  # type: ignore[arg-type]
            omschrijving=waarden.get("omschrijving", ""),
            stadsdeel=waarden.get("stadsdeel", ""),
        )
    return gebieden


def gebied_op_naam(naam: str, bestand: Path | None = None) -> Gebied:
    gebieden = laad_gebieden(bestand)
    if naam in gebieden:
        return gebieden[naam]
    beschikbaar = ", ".join(sorted(gebieden))
    raise KeyError(f"Onbekend gebied '{naam}'. Beschikbaar: {beschikbaar}")


def gebied_uit_bbox(bbox: tuple[float, float, float, float], naam: str = "eigen_gebied") -> Gebied:
    return Gebied(naam=naam, bbox=bbox, omschrijving="Handmatig opgegeven bbox in RD")
