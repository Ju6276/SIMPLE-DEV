"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import importlib
from typing import ClassVar, Type
from simple.core.robot import Robot
from simple.core.registry import RegistryMixin

class RobotRegistry(RegistryMixin[Robot]):

    _MODULE_BY_UID: ClassVar[dict[str, str]] = {
        "franka_fr3": "simple.robots.franka_fr3",
        "aloha": "simple.robots.aloha",
        "vega_1": "simple.robots.vega",
        "g1": "simple.robots.g1",
        "g1_inspire": "simple.robots.g1_inspire",
        "g1_wholebody": "simple.robots.g1_wholebody",
        "g1_inspire_wholebody": "simple.robots.g1_inspire_wholebody",
        "g1_sonic": "simple.robots.g1_sonic",
    }

    @classmethod
    def _base_type(cls) -> Type:
        return Robot

    @classmethod
    def make(cls, uid: str, *args, **kwargs) -> Robot:
        if uid not in cls._registry:
            module_name = cls._MODULE_BY_UID.get(uid)
            if module_name is not None:
                importlib.import_module(module_name)
        return super().make(uid, *args, **kwargs)


    # _registry: ClassVar[dict[str, type[Robot]]] = {}
    # _instances: ClassVar[dict[str, Robot]] = {}

    # @classmethod
    # def register(cls, name: str):
    #     def wrapper(subclass: type[Robot]):
    #         if not issubclass(subclass, Robot):
    #             raise TypeError(f"{subclass.__name__} must inherit from {Robot.__name__}")
    #         cls._registry[name] = subclass
    #         return subclass
    #     return wrapper
    
    # @classmethod
    # def make(cls, res_id: str, *args, **kwargs) -> 'Robot':
    #     """Get the Robot instance for a specific Robot ID."""
        
    #     if res_id not in cls._registry:
    #         raise ValueError(f"No Robot registered for Robot UID '{res_id}'")
        
    #     if res_id not in cls._instances:
    #         # Create and cache the singleton instance
    #         cls._instances[res_id] = cls._registry[res_id](*args, **kwargs)
        
    #     return cls._instances[res_id]
