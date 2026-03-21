from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    CompetitionStartRequest,
    CompetitionStartResponse,
    RunStatusResponse,
    state_to_summary,
)
from workflows.kaggle_workflow import run_kaggle_workflow

app = FastAPI(title="Agentic Kaggle API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_RUNS: Dict[str, Dict[str, Any]] = {}


def _execute_run(run_id: str, competition_ref: str | None, workspace_dir: str | None) -> None:
    _RUNS[run_id]["status"] = "running"
    try:
        result = run_kaggle_workflow(
            competition_ref=competition_ref,
            workspace_dir=workspace_dir,
            run_id=run_id,
            verbose=False,
        )
        summary = state_to_summary(dict(result))
        _RUNS[run_id]["status"] = "done"
        _RUNS[run_id]["result"] = summary
    except Exception as e:
        _RUNS[run_id]["status"] = "error"
        _RUNS[run_id]["error"] = str(e)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/competition/start", response_model=CompetitionStartResponse)
def start_competition(req: CompetitionStartRequest, tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    _RUNS[run_id] = {"status": "pending", "result": None, "error": None}
    tasks.add_task(
        _execute_run,
        run_id,
        req.competition_ref,
        req.workspace_dir,
    )
    return CompetitionStartResponse(run_id=run_id, status="started")


@app.get("/competition/{run_id}/status", response_model=RunStatusResponse)
def competition_status(run_id: str):
    row = _RUNS.get(run_id)
    if not row:
        return RunStatusResponse(run_id=run_id, status="unknown")
    return RunStatusResponse(
        run_id=run_id,
        status=row.get("status", "unknown"),
        result=row.get("result"),
        error=row.get("error"),
    )


@app.websocket("/ws/competition/{run_id}")
async def ws_competition(websocket: WebSocket, run_id: str):
    await websocket.accept()
    row = _RUNS.get(run_id)
    await websocket.send_json(
        {
            "type": "snapshot",
            "payload": {"status": (row or {}).get("status"), "result": (row or {}).get("result")},
        }
    )
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return


def main():
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
