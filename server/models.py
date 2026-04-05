"""Data models for EcoRoute Environment - Advanced Version"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class VehicleType(Enum):
    ELECTRIC = "electric"
    HYBRID = "hybrid"
    DIESEL = "diesel"

class WeatherCondition(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    TRAFFIC = "traffic"

@dataclass
class EcoRouteAction:
    """Action the agent can take - move to a location or wait"""
    next_location_id: int  # -1 = wait/charge, 0=warehouse, 1-4=delivery points
    charge_vehicle: bool = False  # Special action to charge EV
    
@dataclass
class EcoRouteObservation:
    """Rich observation space for the agent"""
    current_location: int
    current_fuel_level: float  # 0.0 to 1.0
    packages_delivered: List[int]
    remaining_packages: List[int]
    packages_with_deadlines: Dict[int, float]  # package_id -> remaining time
    fuel_used: float
    time_elapsed: float
    total_reward_so_far: float
    weather: str
    traffic_level: float  # 0.0 to 1.0
    legal_actions: List[int] = field(default_factory=list)
    is_charging: bool = False
    
@dataclass
class EcoRouteState:
    """Complete state including metadata for evaluation"""
    episode_id: str
    step_count: int
    done: bool
    score: float
    task_level: str
    metrics: Dict[str, Any] = field(default_factory=dict)