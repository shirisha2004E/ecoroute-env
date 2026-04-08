"""Core Environment Logic - Advanced EcoRoute Environment"""
import numpy as np
import uuid
from typing import List, Tuple, Dict, Any
from datetime import datetime
from .models import EcoRouteAction, EcoRouteObservation, EcoRouteState, WeatherCondition

class EcoRouteEnvironment:
    def __init__(self):
        # ========== MAP CONFIGURATION ==========
        # Distance matrix (fuel cost) between locations
        self.distances = {
            0: {1: 5.0, 2: 8.0, 3: 12.0, 4: 15.0, -1: 0.0},  # warehouse
            1: {0: 5.0, 2: 4.0, 3: 7.0, 4: 10.0, -1: 0.0},
            2: {0: 8.0, 1: 4.0, 3: 3.0, 4: 6.0, -1: 0.0},
            3: {0: 12.0, 1: 7.0, 2: 3.0, 4: 4.0, -1: 0.0},
            4: {0: 15.0, 1: 10.0, 2: 6.0, 3: 4.0, -1: 0.0},
        }
        
        # Time windows for deliveries (deadlines in minutes)
        self.deadlines = {
            1: 30,   # Package to location 1 must be delivered within 30 min
            2: 45,   # Package to location 2 within 45 min
            3: 60,   # Package to location 3 within 60 min
            4: 75,   # Package to location 4 within 75 min
        }
        
        # Package values (priority - higher value = more important)
        self.package_values = {
            1: 100,   # High priority
            2: 80,
            3: 60,
            4: 40,    # Low priority
        }
        
        # Vehicle specs
        self.max_fuel = 100.0
        self.charging_rate = 20.0  # Fuel gained per charging step
        self.base_speed = 1.0  # km per minute
        
        self.reset()
    
    def reset(self, task_level: str = "easy", seed: int = None) -> EcoRouteObservation:
        """Reset environment with task-specific configuration"""
        if seed:
            np.random.seed(seed)
        
        # Basic state
        self.current_location = 0  # Start at warehouse
        self.current_fuel = 100.0 if task_level != "hard" else 60.0  # Hard mode: less fuel
        self.fuel_used = 0.0
        self.time_elapsed = 0.0
        self.step_count = 0
        self.task_level = task_level
        self.total_reward = 0.0
        self.is_charging = False
        
        # Set packages based on difficulty
        if task_level == "easy":
            self.packages = [1]
            self.max_steps = 15
            self.traffic_base = 0.2
        elif task_level == "medium":
            self.packages = [1, 2, 3]
            self.max_steps = 25
            self.traffic_base = 0.4
        else:  # hard
            self.packages = [1, 2, 3, 4]
            self.max_steps = 35
            self.traffic_base = 0.6
        
        self.delivered = []
        self.delivery_times = {}  # Track when each package was delivered
        self.failed_deadlines = []
        
        # Weather (random but realistic)
        self.weather = np.random.choice(["clear", "rain", "traffic"], p=[0.6, 0.2, 0.2])
        
        self.done = False
        
        return self._get_observation()
    
    def step(self, action: EcoRouteAction) -> Tuple[EcoRouteObservation, float, bool]:
        """Execute action and compute rich reward"""
        next_loc = action.next_location_id
        self.is_charging = action.charge_vehicle
        
        # ========== VALIDATION ==========
        if next_loc not in self.distances.get(self.current_location, {}):
            return self._get_observation(), -5.0, True  # Invalid move penalty
        
        # ========== CHARGING LOGIC ==========
        if action.charge_vehicle and next_loc == -1:
            # Charging at warehouse
            fuel_gained = min(self.charging_rate, self.max_fuel - self.current_fuel)
            self.current_fuel += fuel_gained
            time_cost = 2.0  # Charging takes time
            self.time_elapsed += time_cost
            reward = 1.0  # Small reward for charging wisely
            self.step_count += 1
            return self._get_observation(), reward, self.done
        
        # ========== MOVEMENT LOGIC ==========
        distance = self.distances[self.current_location][next_loc]
        
        # Traffic effect (higher in hard mode)
        traffic_multiplier = 1.0 + (self.traffic_base * np.random.uniform(0.5, 1.5))
        if self.weather == "traffic":
            traffic_multiplier *= 1.5
        
        # Rain effect
        if self.weather == "rain":
            traffic_multiplier *= 1.3
        
        # Time to travel
        time_cost = (distance / self.base_speed) * traffic_multiplier
        fuel_cost = distance * (1.0 + 0.2 * self.traffic_base)  # More fuel in traffic
        
        # Check if enough fuel
        if fuel_cost > self.current_fuel:
            return self._get_observation(), -10.0, True  # Run out of fuel = game over
        
        # Update state
        self.current_fuel -= fuel_cost
        self.fuel_used += fuel_cost
        self.time_elapsed += time_cost
        self.current_location = next_loc
        
        # ========== REWARD CALCULATION (Advanced) ==========
        reward = 0.0
        
        # 1. Fuel efficiency reward (encourage less fuel usage)
        fuel_efficiency = 1.0 - (fuel_cost / distance) if distance > 0 else 1.0
        reward += fuel_efficiency * 0.5
        
        # 2. Time efficiency penalty
        time_penalty = -0.01 * time_cost
        reward += time_penalty
        
        # 3. Delivery reward
        if next_loc in self.packages and next_loc not in self.delivered:
            self.delivered.append(next_loc)
            self.delivery_times[next_loc] = self.time_elapsed
            
            # Base delivery reward
            delivery_reward = 20.0 * (self.package_values[next_loc] / 100.0)
            reward += delivery_reward
            
            # On-time delivery bonus
            remaining_deadline = self.deadlines[next_loc] - self.time_elapsed
            if remaining_deadline > 0:
                on_time_bonus = 10.0 * (remaining_deadline / self.deadlines[next_loc])
                reward += on_time_bonus
            else:
                self.failed_deadlines.append(next_loc)
                reward -= 15.0  # Penalty for late delivery
            
            # Sequential delivery bonus
            if len(self.delivered) == len(self.packages):
                completion_bonus = 30.0
                reward += completion_bonus
                
                # Speed bonus
                if self.time_elapsed < sum(self.deadlines[p] for p in self.packages) / 2:
                    reward += 20.0
            else:
                # Partial progress reward
                progress_ratio = len(self.delivered) / len(self.packages)
                reward += progress_ratio * 2.0
        
        # 4. Fuel level penalty (encourage not running low)
        if self.current_fuel < 20:
            reward -= 0.5
        
        # 5. Exploration bonus for new locations
        if next_loc not in self.delivered and next_loc != 0:
            reward += 0.1
        
        self.total_reward += reward
        self.step_count += 1
        
        # ========== CHECK DONE CONDITIONS ==========
        if len(self.delivered) == len(self.packages):
            self.done = True
        elif self.current_fuel <= 0:
            reward -= 20.0  # Heavy penalty for running out of fuel
            self.done = True
        elif self.step_count >= self.max_steps:
            self.done = True
        elif self.time_elapsed > 120:  # 2 hour max
            self.done = True
        
        return self._get_observation(), reward, self.done
    
    def _get_observation(self) -> EcoRouteObservation:
        """Build rich observation"""
        remaining = [p for p in self.packages if p not in self.delivered]
        
        # Calculate remaining deadlines
        deadlines_remaining = {}
        for p in remaining:
            remaining_time = max(0, self.deadlines[p] - self.time_elapsed)
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
        """Return full state with metrics"""
        # Calculate final score (strictly between 0 and 1)
        max_possible_score = len(self.packages) * 30.0 + 50.0
        raw_score = self.total_reward + 50.0  # Base score
        # This ensures score is never 0.0 or 1.0 (required by hackathon)
        normalized_score = max(0.01, min(0.99, raw_score / max_possible_score))
        
        # Calculate delivery success rate
        delivery_success_rate = len(self.delivered) / len(self.packages) if self.packages else 0.99
        
        # On-time delivery rate
        on_time_rate = len([p for p in self.delivered if self.delivery_times.get(p, 999) <= self.deadlines.get(p, 999)]) / max(1, len(self.delivered))
        
        metrics = {
            "deliveries_completed": len(self.delivered),
            "total_deliveries": len(self.packages),
            "on_time_rate": on_time_rate,
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
    
    def get_task_grader(self, task_level: str) -> Dict[str, Any]:
        """Return grader for specific task level"""
        if task_level == "easy":
            return {
                "name": "Easy Task: Single Delivery",
                "pass_criteria": lambda s: len(s.delivered) == 1,
                "score_criteria": lambda s: 0.5 if len(s.delivered) == 1 else 0.0
            }
        elif task_level == "medium":
            return {
                "name": "Medium Task: 3 Deliveries",
                "pass_criteria": lambda s: len(s.delivered) >= 2,
                "score_criteria": lambda s: len(s.delivered) / 3.0
            }
        else:
            return {
                "name": "Hard Task: 4 Deliveries with Deadlines",
                "pass_criteria": lambda s: len(s.delivered) >= 3,
                "score_criteria": lambda s: (len(s.delivered) / 4.0) * (s.state().metrics["on_time_rate"])
            }