from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from .environment import EcoRouteEnvironment
from .models import EcoRouteAction, EcoRouteObservation, EcoRouteState

# Create environment instance
env = EcoRouteEnvironment()

app = FastAPI(
    title="EcoRoute Environment API",
    description="Real-world delivery route optimization for AI agents",
    version="2.0.0"
)

# Enable CORS for Hugging Face Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResetRequest(BaseModel):
    task_level: str = "easy"
    seed: int = None

class StepRequest(BaseModel):
    action: EcoRouteAction

@app.get("/")
async def root():
    return {
        "message": "EcoRoute Environment API",
        "version": "2.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks"]
    }

@app.post("/reset")
async def reset_endpoint(request: ResetRequest = None):
    """Reset the environment for a new episode"""
    task_level = request.task_level if request else "easy"
    seed = request.seed if request else None
    observation = env.reset(task_level=task_level, seed=seed)
    return {
        "observation": observation.__dict__,
        "reward": 0.0,
        "done": False,
        "task_level": task_level
    }

@app.post("/step")
async def step_endpoint(request: StepRequest):
    """Take an action and get result"""
    observation, reward, done = env.step(request.action)
    return {
        "observation": observation.__dict__,
        "reward": reward,
        "done": done
    }

@app.get("/state")
async def state_endpoint():
    """Get current environment state"""
    state = env.state()
    return {
        "episode_id": state.episode_id,
        "step_count": state.step_count,
        "done": state.done,
        "score": state.score,
        "task_level": state.task_level,
        "metrics": state.metrics
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": "ecoroute"}

@app.get("/tasks")
async def list_tasks():
    """List available tasks and their requirements"""
    return {
        "easy": {
            "description": "Deliver 1 package to location 1",
            "max_steps": 15,
            "deadline_minutes": 30
        },
        "medium": {
            "description": "Deliver 3 packages avoiding traffic",
            "max_steps": 25,
            "deadline_minutes": 45
        },
        "hard": {
            "description": "Deliver 4 packages with time windows and eco-score",
            "max_steps": 35,
            "deadline_minutes": 75
        }
    }

# ========== ADD THIS AT THE BOTTOM ==========
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()