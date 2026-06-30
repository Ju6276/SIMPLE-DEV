"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FrankaResearch3",
    "Aloha",
    "Vega1",
    "WristCamMountable",
    "G1",
    "G1Inspire",
    "G1Wholebody",
    "G1InspireWholebody",
    "G1Sonic",
]

_MODULE_BY_NAME = {
    "FrankaResearch3": ("simple.robots.franka_fr3", "FrankaResearch3"),
    "Aloha": ("simple.robots.aloha", "Aloha"),
    "Vega1": ("simple.robots.vega", "Vega1"),
    "WristCamMountable": ("simple.robots.protocols", "WristCamMountable"),
    "G1": ("simple.robots.g1", "G1"),
    "G1Inspire": ("simple.robots.g1_inspire", "G1Inspire"),
    "G1Wholebody": ("simple.robots.g1_wholebody", "G1Wholebody"),
    "G1InspireWholebody": ("simple.robots.g1_inspire_wholebody", "G1InspireWholebody"),
    "G1Sonic": ("simple.robots.g1_sonic", "G1Sonic"),
}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _MODULE_BY_NAME[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
