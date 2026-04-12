"""Core Environment Logic - Advanced EcoRoute Environment"""
import numpy as np
import uuid
import random
from typing import List, Tuple, Dict, Any
from datetime import datetime
from .models import EcoRouteAction, EcoRouteObservation, EcoRouteState, WeatherCondition

class EcoRouteEnvironment:
    def __init__(self):
        # ========== MAP CONFIGURATION ==========
        # Distance matrix (fuel cost) between locations
        self.distances = {
            0: {1: 5.0, 2: 8.0, 3: 12.0, 4: 15.0, 5: 18.0, 6: 20.0, -1: 0.0},  # warehouse + more locations
            1: {0: 5.0, 2: 4.0, 3: 7.0, 4: 10.0, 5: 13.0, 6: 16.0, -1: 0.0},
            2: {0: 8.0, 1: 4.0, 3: 3.0, 4: 6.0, 5: 9.0, 6: 12.0, -1: 0.0},
            3: {0: 12.0, 1: 7.0, 2: 3.0, 4: 4.0, 5: 7.0, 6: 10.0, -1: 0.0},
            4: {0: 15.0, 1: 10.0, 2: 6.0, 3: 4.0, 5: 5.0, 6: 8.0, -1: 0.0},
            5: {0: 18.0, 1: 13.0, 2: 9.0, 3: 7.0, 4: 5.0, 6: 3.0, -1: 0.0},
            6: {0: 20.0, 1: 16.0, 2: 12.0, 3: 10.0, 4: 8.0, 5: 3.0, -1: 0.0},
        }
        
        # Time windows for deliveries
        self.time_windows = {
            1: {"start": 0, "end": 30},
            2: {"start": 10, "end": 45},
            3: {"start": 20, "end": 60},
            4: {"start": 30, "end": 75},
            5: {"start": 40, "end": 90},
            6: {"start": 50, "end": 105},
        }
        
        # Package values
        self.package_values = {1: 100, 2: 90, 3: 80, 4: 70, 5: 60, 6: 50}
        
        self.max_fuel = 100.0
        self.charging_rate = 25.0
        self.base_speed = 1.0
        
        self.reset()
    
    def reset(self, task_level: str = "easy", seed: int = None) -> EcoRouteObservation:
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_location = 0
        self.current_fuel = 100.0
        self.fuel_used = 0.0
        self.time_elapsed = 0.0
        self.step_count = 0
        self.task_level = task_level
        self.total_reward = 0.0
        self.is_charging = False
        
        # Enhanced task difficulty
        if task_level == "easy":
            self.packages = [1, 2]
            self.max_steps = 20
            self.traffic_base = 0.2
        elif task_level == "medium":
            self.packages = [1, 2, 3, 4]
            self.max_steps = 30
            self.traffic_base = 0.4
        else:  # hard
            self.packages = [1, 2, 3, 4, 5, 6]
            self.max_steps = 45
            self.traffic_base = 0.6
        
        self.delivered = []
        self.delivery_times = {}
        self.missed_windows = []
        
        # Dynamic weather that can change during episode
        self.weather = random.choice(["clear", "rain", "traffic"])
        self.weather_change_counter = 0
        
        self.done = False
        
        return self._get_observation()
    
    def step(self, action: EcoRouteAction) -> Tuple[EcoRouteObservation, float, bool]:
        next_loc = action.next_location_id
        self.is_charging = action.charge_vehicle
        
        # Change weather every 5 steps
        self.weather_change_counter += 1
        if self.weather_change_counter >= 5:
            self.weather = random.choice(["clear", "rain", "traffic"])
            self.weather_change_counter = 0
        
        if next_loc not in self.distances.get(self.current_location, {}):
            return self._get_observation(), -10.0, True
        
        # Charging logic
        if action.charge_vehicle and next_loc == -1:
            fuel_gained = min(self.charging_rate, self.max_fuel - self.current_fuel)
            self.current_fuel += fuel_gained
            time_cost = 2.0
            self.time_elapsed += time_cost
            reward = 2.0 if fuel_gained > 0 else -1.0
            self.step_count += 1
            return self._get_observation(), reward, self.done
        
        distance = self.distances[self.current_location][next_loc]
        
        # Dynamic traffic multiplier
        traffic_multiplier = 1.0 + (self.traffic_base * np.random.uniform(0.5, 1.5))
        if self.weather == "traffic":
            traffic_multiplier *= 1.8
        elif self.weather == "rain":
            traffic_multiplier *= 1.4
        
        time_cost = (distance / self.base_speed) * traffic_multiplier
        fuel_cost = distance * (1.0 + 0.3 * self.traffic_base)
        
        if fuel_cost > self.current_fuel:
            return self._get_observation(), -20.0, True
        
        self.current_fuel -= fuel_cost
        self.fuel_used += fuel_cost
        self.time_elapsed += time_cost
        self.current_location = next_loc
        
        # Enhanced reward calculation
        reward = 0.0
        
        # Distance efficiency
        fuel_efficiency = 1.0 - (fuel_cost / distance) if distance > 0 else 1.0
        reward += fuel_efficiency * 1.0
        
        # Time penalty
        reward -= 0.02 * time_cost
        
        # Delivery logic
        if next_loc in self.packages and next_loc not in self.delivered:
            self.delivered.append(next_loc)
            self.delivery_times[next_loc] = self.time_elapsed
            
            # Time window check
            window = self.time_windows.get(next_loc, {"start": 0, "end": 999})
            in_window = window["start"] <= self.time_elapsed <= window["end"]
            
            delivery_reward = 30.0 * (self.package_values[next_loc] / 100.0)
            reward += delivery_reward
            
            if in_window:
                reward += 25.0  # Big bonus for time window compliance
            else:
                self.missed_windows.append(next_loc)
                reward -= 20.0  # Penalty for missing window
            
            # Completion bonus
            if len(self.delivered) == len(self.packages):
                completion_bonus = 50.0
                reward += completion_bonus
                
                # Speed bonus
                if self.time_elapsed < len(self.packages) * 15:
                    reward += 30.0
                
                # Fuel efficiency bonus
                if self.fuel_used < len(self.packages) * 8:
                    reward += 25.0
            else:
                progress_ratio = len(self.delivered) / len(self.packages)
                reward += progress_ratio * 5.0
        
        # Penalties
        if self.current_fuel < 15:
            reward -= 2.0
        
        if next_loc not in self.delivered and next_loc != 0:
            reward += 0.5  # Exploration bonus
        
        self.total_reward += reward
        self.step_count += 1
        
        # Done conditions
        if len(self.delivered) == len(self.packages):
            self.done = True
        elif self.current_fuel <= 0:
            reward -= 30.0
            self.done = True
        elif self.step_count >= self.max_steps:
            self.done = True
        elif self.time_elapsed > 120:
            self.done = True
        
        return self._get_observation(), reward, self.done
    
    def _get_observation(self) -> EcoRouteObservation:
        remaining = [p for p in self.packages if p not in self.delivered]
        
        deadlines_remaining = {}
        for p in remaining:
            window = self.time_windows.get(p, {"end": 999})
            remaining_time = max(0, window["end"] - self.time_elapsed)
            deadlines_remaining[p] = remaining_time
        
        return EcoRouteObservation(
            current_location=self.current_location,
            current_fuel_level=self.current_fuel / self.max_fuel,
            packages_delivered=self.delivered.copy(),
            remaining_packages=remaining,
            packages_with_deadlines=deadlines_remaining,
            fuel_used=self.fuel_used,
            time_elapsed=self.time_elapsed,
            total_reward_so_far=self.total_reward,
            weather=self.weather,
            traffic_level=self.traffic_base,
            legal_actions=list(self.distances.get(self.current_location, {}).keys()),
            is_charging=self.is_charging
        )
    
    def state(self) -> EcoRouteState:
        max_possible_score = len(self.packages) * 50.0 + 100.0
        raw_score = self.total_reward + 100.0
        normalized_score = max(0.01, min(0.99, raw_score / max_possible_score))
        
        delivery_success_rate = len(self.delivered) / len(self.packages) if self.packages else 0.99
        
        window_compliance_rate = len([p for p in self.delivered if p not in self.missed_windows]) / max(1, len(self.delivered))
        
        metrics = {
            "deliveries_completed": len(self.delivered),
            "total_deliveries": len(self.packages),
            "window_compliance_rate": window_compliance_rate,
            "fuel_efficiency": (self.fuel_used / max(1, len(self.delivered))) if self.delivered else 0,
            "total_time": self.time_elapsed,
            "success_rate": delivery_success_rate,
        }
        
        return EcoRouteState(
            episode_id=f"eco_{self.task_level}_{uuid.uuid4().hex[:6]}",
            step_count=self.step_count,
            done=self.done,
            score=normalized_score,
            task_level=self.task_level,
            metrics=metrics
        )