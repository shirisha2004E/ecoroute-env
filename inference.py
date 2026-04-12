"""Baseline Inference Script with Advanced LLM Agent"""
import asyncio
import sys
import os
from typing import Dict, Any
import httpx
from openai import OpenAI

def log_start(task_level: str):
    print(f"[START] task={task_level}", flush=True)

def log_step(step_num: int, reward: float):
    print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)

def log_end(final_score: float):
    print(f"[END] task= score={final_score:.4f} steps=", flush=True)

class AdvancedLLMAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("API_KEY", ""),
        )
        self.model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    
    def get_action(self, observation: Dict) -> Dict:
        remaining = observation.get("remaining_packages", [])
        current = observation.get("current_location", 0)
        fuel_level = observation.get("current_fuel_level", 1.0)
        time_elapsed = observation.get("time_elapsed", 0)
        deadlines = observation.get("packages_with_deadlines", {})
        
        if not remaining:
            return {"next_location_id": 0}
        
        # Build smart prompt
        prompt = f"""You are a delivery optimizer. Current location: {current}. Time: {time_elapsed:.1f}. Fuel: {fuel_level:.0%}.
Remaining deliveries: {remaining}. Deadlines (remaining time): {deadlines}.
Choose the BEST next delivery location to maximize on-time deliveries and fuel efficiency.
Consider urgency (short deadlines), distance, and fuel level.
Return ONLY the location number."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=10
            )
            next_loc = int(response.choices[0].message.content.strip())
            if next_loc in remaining:
                return {"next_location_id": next_loc}
        except:
            pass
        
        # Fallback: choose most urgent (shortest deadline)
        if deadlines:
            urgent = min(deadlines, key=deadlines.get)
            return {"next_location_id": urgent}
        
        return {"next_location_id": remaining[0]}

async def run_baseline(env_url: str, task: str = "easy"):
    log_start(task)
    agent = AdvancedLLMAgent()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        reset_response = await client.post(f"{env_url}/reset", params={"task_level": task})
        reset_data = reset_response.json()
        
        step_num = 0
        total_reward = 0.0
        done = False
        observation = reset_data.get("observation", {})
        
        while not done and step_num < 60:
            action = agent.get_action(observation)
            step_response = await client.post(f"{env_url}/step", json={"action": action})
            step_data = step_response.json()
            
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            observation = step_data.get("observation", {})
            total_reward += reward
            
            log_step(step_num, reward)
            step_num += 1
        
        state_response = await client.get(f"{env_url}/state")
        state = state_response.json()
        final_score = state.get("score", 0.0)
        
        log_end(final_score)
        return final_score

async def run_all_tasks():
    space_url = "https://EShirisha630-ecorouteenv.hf.space"
    
    for task in ["easy", "medium", "hard"]:
        print(f"\nRunning {task.upper()} task...")
        score = await run_baseline(env_url=space_url, task=task)
        print(f"✅ {task.upper()} - Score: {score:.4f}\n")

if __name__ == "__main__":
    asyncio.run(run_all_tasks())
    sys.exit(0)