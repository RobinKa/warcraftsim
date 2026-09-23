"""warcraftsim: headless Warcraft III (Legacy TFT 1.29) under Wine, controlled from Python."""

from .client import Wc3Game
from .protocol import Observation, PlayerState, Result, Unit, UnitFlags
from .runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup, Idle, Scripted, Slot
from .scenario import Scenario

__all__ = [
    "Agent", "BuiltinAI", "GameInstance", "GameSetup", "Idle", "Observation", "PlayerState", "Result", "Scenario",
    "Scripted", "Slot", "Unit", "UnitFlags", "Wc3Game",
]
