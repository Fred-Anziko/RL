import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .baa_interface import BAAAgent

app = FastAPI()

class Observation(BaseModel):
    state: List[float]
    rtg: float
    session_id: str = "default"
    reward: Optional[float] = 0.0
    done: Optional[bool] = False

# Task-specific config moved to environment variables or a config file
agent = BAAAgent(
    state_dim=int(os.getenv("STATE_DIM", 16)), 
    action_dim=int(os.getenv("ACTION_DIM", 4))
)

@app.post("/predict")
async def predict(obs: Observation):
    try:
        action, trace, curiosity = agent.get_action(obs.state, obs.rtg, session_id=obs.session_id)
        
        # Background training logic...
        agent.record_experience(obs.session_id, obs.state, action, obs.reward, obs.done)
        
        return {
            "action": action.tolist(),
            "curiosity": float(curiosity)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))