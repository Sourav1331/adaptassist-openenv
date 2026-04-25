"""
AdaptAssist HF Space server v2.

Endpoints (OpenEnv-compliant):
  GET  /          → frontend UI
  GET  /info      → environment metadata
  GET  /health    → health check
  GET  /tasks     → list available task types
  GET  /state     → current episode state
  GET  /metrics   → aggregate episode stats
  POST /reset     → start new episode
  POST /step      → take one action
"""

import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import AdaptAssistEnv

app = FastAPI(
    title="AdaptAssist Environment",
    description="Drift-aware personal AI agent environment for OpenEnv hackathon.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ENV: Optional[AdaptAssistEnv] = None

TASK_TYPES = ["resolve_conflict", "reply_email", "book_restaurant", "reschedule", "prioritise_day"]
DIFF_MAP = {"easy": 1, "medium": 2, "hard": 3}


class ResetRequest(BaseModel):
    difficulty: str = "easy"
    seed: Optional[int] = None
    n_tasks: int = 3


class StepRequest(BaseModel):
    tool: str
    params: dict = Field(default_factory=dict)
    thought: Optional[str] = None


def _meta() -> dict:
    return {
        "name": "AdaptAssist Environment",
        "version": "2.1.0",
        "benchmark_id": "adaptassist_drift",
        "theme": "World Modeling — Personalized Tasks (#3.2)",
        "author": "Souravdanyal",
        "tools": AdaptAssistEnv.TOOLS,
        "difficulties": list(DIFF_MAP.keys()),
        "task_types": TASK_TYPES,
        "endpoints": {
            "GET /": "Frontend UI",
            "GET /info": "Environment metadata",
            "GET /health": "Health check",
            "GET /tasks": "Available task types",
            "POST /reset": "Start new episode",
            "POST /step": "Take one action",
            "GET /state": "Current episode state",
            "GET /metrics": "Aggregate episode stats",
        },
    }


@app.get("/")
def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return _meta()


@app.get("/info")
def info():
    return _meta()


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.1.0", "env_ready": ENV is not None}


@app.get("/tasks")
def tasks():
    return {"task_types": TASK_TYPES, "difficulties": list(DIFF_MAP.keys())}


@app.post("/reset")
def reset(req: ResetRequest):
    global ENV
    if req.difficulty not in DIFF_MAP:
        raise HTTPException(400, f"difficulty must be one of {list(DIFF_MAP.keys())}")
    n = max(1, min(5, req.n_tasks))
    ENV = AdaptAssistEnv(difficulty=DIFF_MAP[req.difficulty], max_steps=30,
                         n_tasks=n, seed=req.seed)
    obs = ENV.reset()
    return {"observation": obs, "difficulty": req.difficulty}


@app.post("/step")
def step(req: StepRequest):
    global ENV
    if ENV is None:
        raise HTTPException(400, "Call /reset first to start an episode.")
    action = {"tool": req.tool, "params": req.params}
    if req.thought:
        action["thought"] = req.thought
    obs, reward, done, info = ENV.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "tool_result": info.get("tool_result", {}),
        "tasks_completed": info.get("tasks_completed", 0),
        "total_tasks": info.get("total_tasks", 0),
        "episode_summary": info.get("episode_summary"),
    }


@app.get("/state")
def state():
    if ENV is None:
        return {"status": "idle", "message": "No active episode. Call /reset first."}
    return {"status": "active", "observation": ENV.observation(), "summary": ENV.state()}


@app.get("/metrics")
def metrics():
    if ENV is None:
        return {"status": "no_episodes_yet"}
    return ENV.get_episode_summary()


if __name__ == "__main__":
    import uvicorn
    is_space = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
    host = "0.0.0.0" if is_space else "127.0.0.1"
    port = int(os.getenv("PORT", os.getenv("APP_PORT", "7860")))
    uvicorn.run(app, host=host, port=port)