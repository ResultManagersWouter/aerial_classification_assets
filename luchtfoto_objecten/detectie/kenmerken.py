from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import rgb2hsv


def naar_float(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    return rgb.astype(np.float32)


def exces_groen(rgb: np.ndarray) -> np.ndarray:
    """Excess green index, het standaard groensignaal voor luchtfoto's zonder infrarood."""
    beeld = naar_float(rgb)
    som = beeld.sum(axis=2)
    som[som == 0] = 1e-6
    rood, groen, blauw = beeld[:, :, 0] / som, beeld[:, :, 1] / som, beeld[:, :, 2] / som
    return 2.0 * groen - rood - blauw


def vari(rgb: np.ndarray) -> np.ndarray:
    beeld = naar_float(rgb)
    rood, groen, blauw = beeld[:, :, 0], beeld[:, :, 1], beeld[:, :, 2]
    noemer = groen + rood - blauw
    noemer = np.where(np.abs(noemer) < 1e-6, 1e-6, noemer)
    return np.clip((groen - rood) / noemer, -1.0, 1.0)


def hsv_kanalen(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = rgb2hsv(naar_float(rgb))
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def grijswaarde(rgb: np.ndarray) -> np.ndarray:
    beeld = naar_float(rgb)
    return 0.299 * beeld[:, :, 0] + 0.587 * beeld[:, :, 1] + 0.114 * beeld[:, :, 2]


def lokale_standaarddeviatie(kanaal: np.ndarray, venster_px: int) -> np.ndarray:
    """Textuurmaat, hoog bij boomkronen en daken, laag bij asfalt en gras."""
    venster_px = max(3, venster_px | 1)
    gemiddelde = uniform_filter(kanaal, size=venster_px)
    gemiddelde_kwadraat = uniform_filter(kanaal * kanaal, size=venster_px)
    variantie = np.maximum(gemiddelde_kwadraat - gemiddelde * gemiddelde, 0.0)
    return np.sqrt(variantie)


def schaduwmasker(rgb: np.ndarray, percentiel: float = 22.0) -> np.ndarray:
    """Donkere pixels met een blauwzweem, typisch voor schaduw onder open hemel."""
    _, verzadiging, waarde = hsv_kanalen(rgb)
    drempel = float(np.percentile(waarde, percentiel))
    beeld = naar_float(rgb)
    blauwzweem = beeld[:, :, 2] >= beeld[:, :, 0] - 0.02
    return (waarde <= drempel) & blauwzweem & (verzadiging < 0.55)
