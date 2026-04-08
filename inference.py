"""Baseline Inference Script with Required Logging Format"""
import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any
import httpx

# ========== REQUIRED LOGGING FUNCTIONS FOR HACKATHON ==========
def log_start(environment_name: str, task_level: str):
    """Emit START log in required format"""
    print(f"[START] environment={environment_name} task={task_level}", flush=True)

def log_step(step_num: int, action: Dict, reward: float, done: bool, observation: Dict):
    """Emit STEP log in required format"""
    packages_left = len(observation.get('remaining_packages', []))
    print(f"[STEP] step={step_num} action={action.get('next_location_id')} reward={reward:.4f} done={done} packages_left={packages_left}", flush=True)

def log_end(total_reward: float, final_score: float, metrics: Dict):
    """Emit END log in required format"""
    print(f"[END] total_reward={total_reward:.4f} score={final_score:.4f}", flush=True)

# ========== ADVANCED BASELINE AGENTS ==========
class GreedyBaselineAgent:
    """Greedy agent that always goes to nearest undelivered package"""
    
    def __init__(self):
        self.name = "greedy"
    
    def get_action(self, observation: Dict) -> Dict:
        remaining = observation.get("remaining_packages", [])
        
        if remaining:
            return {"next_location_id": remaining[0]}
        
        return {"next_location_id": 0}

class SmartHeuristicAgent:
    """Smart agent considering deadlines, fuel, and distance"""
    
    def __init__(self):
        self.name = "smart_heuristic"
    
    def get_action(self, observation: Dict) -> Dict:
        remaining = observation.get("remaining_packages", [])
        deadlines = observation.get("packages_with_deadlines", {})
        fuel_level = observation.get("current_fuel_level", 1.0)
        
        if not remaining:
            return {"next_location_id": 0}
        
        best_action = None
        best_score = -float('inf')
        legal_actions = observation.get("legal_actions", remaining + [0])
        
        for action in legal_actions:
            score = 0
            
            if action in remaining:
                urgency = deadlines.get(action, 999)
                if urgency < 10:
                    score += 100
                elif urgency < 30:
                    score += 50
                else:
                    score += 30
                
                if action == remaining[0]:
                    score += 10
            elif action == 0:
                if fuel_level < 0.3:
                    score += 40
                else:
                    score += 5
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return {"next_location_id": best_action if best_action is not None else 0}

# ========== MAIN BASELINE FUNCTION ==========
async def run_baseline(env_url: str = None, task: str = "easy", agent_type: str = "smart"):
    """Run baseline agent and return score"""
    
    # Use your Space URL
    if env_url is None:
        env_url = "https://EShirisha630-ecorouteenv.hf.space"
    
    log_start("ecoroute-env", task)
    
    # Select agent
    if agent_type == "greedy":
        agent = GreedyBaselineAgent()
    else:
        agent = SmartHeuristicAgent()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Reset environment
        reset_response = await client.post(f"{env_url}/reset", params={"task_level": task})
        reset_data = reset_response.json()
        
        step_num = 0
        total_reward = 0.0
        done = False
        observation = reset_data.get("observation", {})
        
        while not done and step_num < 50:
            # Get action from agent
            action = agent.get_action(observation)
            
            # Take step
            step_response = await client.post(f"{env_url}/step", json={"action": action})
            step_data = step_response.json()
            
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            observation = step_data.get("observation", {})
            total_reward += reward
            
            log_step(step_num, action, reward, done, observation)
            
            step_num += 1
        
        # Get final state
        state_response = await client.get(f"{env_url}/state")
        state = state_response.json()
        
        log_end(total_reward, state.get("score", 0.0), state.get("metrics", {}))
        
        return {
            "task": task,
            "total_reward": total_reward,
            "final_score": state.get("score", 0.0),
            "steps": step_num,
            "metrics": state.get("metrics", {})
        }

async def run_all_tasks():
    """Run baseline on all three tasks"""
    results = {}
    
    space_url = "https://EShirisha630-ecorouteenv.hf.space"
    
    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Running {task.upper()} task...")
        print(f"{'='*50}\n")
        
        result = await run_baseline(env_url=space_url, task=task, agent_type="smart")
        results[task] = result
        
        print(f"\n✅ {task.upper()} - Score: {result['final_score']:.4f}, Reward: {result['total_reward']:.2f}\n")
    
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for task, result in results.items():
        print(f"{task.upper()}: Score = {result['final_score']:.4f}")
    
    avg_score = sum(r['final_score'] for r in results.values()) / len(results)
    print(f"\nAverage Score: {avg_score:.4f}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_all_tasks())
    sys.exit(0)