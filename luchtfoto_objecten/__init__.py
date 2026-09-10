"""Objecten herkennen op luchtfoto's en vergelijken met de gemeentelijke registratie."""

from luchtfoto_objecten.gebieden import Gebied, gebied_op_naam, laad_gebieden
from luchtfoto_objecten.instellingen import Instellingen

__all__ = ["Gebied", "Instellingen", "gebied_op_naam", "laad_gebieden"]
__version__ = "0.1.0"
