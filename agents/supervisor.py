"""Supervisor agent for coordinating data processing, training, and validation with Kaggle integration"""

import json
import os

import pandas as pd
from pathlib import Path
from typing import Dict, Any, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from utils import logger
from utils.session_manager import SessionManager
from utils.kaggle_utils import KaggleManager, extract_json_from_response
from agents.data_worker import run_data_worker, DataWorkState
from agents.trainer import run_trainer
from agents.validator import run_validator

class SupervisorState(TypedDict):
    competition_name: str
    kaggle_manager: Optional[KaggleManager]
    competition_info: Dict[str, Any]
    session: Optional[SessionManager]
    data_worker_result: Dict[str, Any]
    trainer_result: Dict[str, Any]
    validator_result: Dict[str, Any]
    submission_result: Dict[str, Any]
    current_phase: str  # 'download', 'analyze', 'data', 'train', 'validate', 'submit', 'feedback'
    iteration: int
    max_iterations: int
    feedback: str
    final_report: str
    last_score: Optional[float]
    improvement_needed: bool
    done: bool

llm = ChatOllama(
    model="qwen2.5-coder:14b-instruct-q4_K_M",
    temperature=0,
)

def download_competition_data(state: SupervisorState) -> Dict[str, Any]:
    """Download competition data from Kaggle"""

    logger.info(f"Downloading competition data: {state['competition_name']}")

    kaggle = KaggleManager(state['competition_name'])

    # Get competition info
    competition_info = kaggle.get_competition_info()

    # Create session
    session = SessionManager(source_dir="data")

    return {
        "kaggle_manager": kaggle,
        "competition_info": competition_info,
        "session": session,
        "current_phase": "analyze"
    }

def analyze_competition(state: SupervisorState) -> Dict[str, Any]:
    """Analyze competition requirements and create strategy"""

    info = state.get("competition_info", {})

    prompt = f"""
You are a Kaggle competition expert. Analyze this competition:

Competition: {state['competition_name']}
Title: {info.get('title', 'Unknown')}
Description: {info.get('description', 'No description')[:1000]}
Evaluation Metric: {info.get('evaluation_metric', 'Unknown')}

Leaderboard (top 5):
{json.dumps(info.get('current_leaderboard', []), indent=2)}

Based on this information:
1. What type of problem is it?
2. What are the key challenges?
3. What approach would you recommend?
4. What's a realistic target score?

Output in JSON format:
{{
    "problem_type": "regression/classification",
    "metric": "metric_name",
    "target_column": "target",
    "key_challenges": ["challenge1", "challenge2"],
    "recommended_approach": "detailed approach description",
    "baseline_score": 0.0,
    "target_score": 0.0
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)]).content
    analysis = extract_json_from_response(response)

    logger.info(f"Competition analysis: {analysis}")

    return {
        "current_phase": "data",
        "competition_info": {**info, **analysis}
    }

def run_data_phase(state: SupervisorState) -> Dict[str, Any]:
    """Run data processing phase with Kaggle context"""

    logger.info("Starting data processing phase...")

    # Get competition info for data processing
    comp_info = state.get("competition_info", {})

    enhanced_task = f"""
Process data for Kaggle competition: {state['competition_name']}

Competition type: {comp_info.get('problem_type', 'regression')}
Target column: {comp_info.get('target_column', 'target')}
Evaluation metric: {comp_info.get('metric', 'mse')}

Key challenges identified:
{chr(10).join([f"- {c}" for c in comp_info.get('key_challenges', [])])}

Recommended approach:
{comp_info.get('recommended_approach', '')}

Focus on:
1. Proper handling of train/test split
2. Feature engineering that matches competition needs
3. Creating submission-ready test predictions format
"""

    data_state = DataWorkState(
        task=enhanced_task,
        session=state["session"],
        data_dir=str(state["session"].session_dir),
        path_to_comp_desc=f"data/competition_info.txt",
        name_of_file="train.csv",
        current_plan=None,
        analytic_attempts=0,
        analytic_max_attempts=4,
        satisfy_rate=0.0,
        worker_attempts=0,
        worker_max_attempts=3,
        previous_results=[],
        done=False
    )

    # Save competition info to file
    with open(state["session"].session_dir / "competition_info.json", 'w') as f:
        json.dump(comp_info, f, indent=2)

    result = run_data_worker(data_state)

    return {
        "data_worker_result": result,
        "current_phase": "train" if result.get("satisfy_rate", 0) > 0.7 else "data"
    }


def run_training_phase(state: SupervisorState) -> Dict[str, Any]:
    """Run model training phase"""

    logger.info("Starting training phase...")

    # Get competition analysis or use defaults
    comp_info = state.get("competition_info", {})

    # Ensure target column is correct
    target_column = comp_info.get("target_column", "target")  # <-- Явно указываем 'target'
    metric = comp_info.get("metric", "mse")

    # Run trainer with explicit target column
    result = run_trainer(
        session=state["session"],
        target_column=target_column,  # <-- Передаем правильное имя
        metric=metric,
        model_type="auto",
        max_attempts=3
    )

    return {
        "trainer_result": result,
        "current_phase": "validate"
    }


def run_validation_phase(state: SupervisorState) -> Dict[str, Any]:
    """Run validation phase"""

    logger.info("Starting validation phase...")

    comp_info = state.get("competition_info", {})

    # Явно указываем целевую колонку
    target_column = "target"  # <-- жестко задаем
    metric = comp_info.get("metric", "mse")

    # Устанавливаем порог - можно сделать адаптивным
    threshold = 15000  # для MSE, можно настроить

    result = run_validator(
        session=state["session"],
        target_column=target_column,
        metric=metric,
        threshold=threshold
    )

    # Проверяем, нужно ли улучшение
    passed = result.get("passed", False)
    score = result.get("scores", {}).get(metric, float('inf'))

    improvement_needed = not passed

    return {
        "validator_result": result,
        "last_score": score,
        "improvement_needed": improvement_needed,
        "current_phase": "submit"
    }


def submit_to_kaggle(state: SupervisorState) -> Dict[str, Any]:
    """Submit predictions to Kaggle"""

    logger.info("Submitting to Kaggle...")

    kaggle = state.get("kaggle_manager")
    if not kaggle:
        return {"current_phase": "feedback", "submission_result": {"error": "No Kaggle manager"}}

    # Find predictions file
    session_dir = state["session"].session_dir
    pred_files = list(session_dir.glob("*prediction*.csv")) + list(session_dir.glob("submission*.csv"))

    if not pred_files:
        # Create submission file from predictions
        predictions = state["trainer_result"].get("predictions")
        if predictions is not None:
            sample_sub = kaggle.download_sample_submission()
            if sample_sub and os.path.exists(sample_sub):
                sample_df = pd.read_csv(sample_sub)
                submission_df = sample_df.copy()
                submission_df.iloc[:, 1] = predictions.flatten()
                submission_path = session_dir / "submission.csv"
                submission_df.to_csv(submission_path, index=False)
                pred_files = [submission_path]

    if not pred_files:
        return {
            "current_phase": "feedback",
            "submission_result": {"error": "No prediction file found"}
        }

    # Submit best predictions
    submission_file = pred_files[0]
    result = kaggle.submit_prediction(str(submission_file))

    submission_result = {
        "success": result[0],
        "submission_id": result[1],
        "file": str(submission_file)
    }

    if result[0]:
        # Get score after submission
        import time
        time.sleep(5)  # Wait for score to update
        score = kaggle.get_last_submission_score()
        submission_result["score"] = score

        # Analyze result
        analysis = kaggle.analyze_submission_result(result[1])
        submission_result["analysis"] = analysis

        # Check if we need improvement
        target_score = state.get("competition_info", {}).get("target_score")
        metric = state.get("competition_info", {}).get("metric", "mse")

        if score is not None and target_score is not None:
            improvement_needed = (score > target_score if metric in ['mse', 'rmse', 'mae']
                                 else score < target_score)
        else:
            improvement_needed = False

        return {
            "submission_result": submission_result,
            "last_score": score,
            "improvement_needed": improvement_needed,
            "current_phase": "analyze_result"
        }

    return {
        "submission_result": submission_result,
        "current_phase": "feedback"
    }

def analyze_kaggle_result(state: SupervisorState) -> Dict[str, Any]:
    """Analyze Kaggle submission result and plan improvements"""

    submission = state.get("submission_result", {})
    score = submission.get("score")
    analysis = submission.get("analysis", {})

    comp_info = state.get("competition_info", {})
    target_score = comp_info.get("target_score")
    metric = comp_info.get("metric", "mse")

    # Get leaderboard for context
    kaggle = state.get("kaggle_manager")
    leaderboard = kaggle.get_leaderboard(limit=10) if kaggle else []

    prompt = f"""
You are a Kaggle competition expert. Analyze the submission result:

Competition: {state['competition_name']}
Metric: {metric}
Your score: {score}
Target score: {target_score}
Leaderboard top: {leaderboard[:3] if leaderboard else 'Unknown'}

Submission analysis: {analysis}

Based on this:
1. Is the performance acceptable?
2. What improvements are needed?
3. Should we try different approaches?
4. Specific recommendations for next iteration

Output in JSON format:
{{
    "acceptable": true/false,
    "improvement_needed": true/false,
    "recommendations": ["rec1", "rec2"],
    "next_strategy": "strategy description",
    "confidence": 0.0-1.0
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)]).content
    feedback = extract_json_from_response(response)

    improvement_needed = feedback.get("improvement_needed", False)
    acceptable = feedback.get("acceptable", False)

    # Update iteration counter
    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", 5)

    if iteration >= max_iterations or acceptable:
        return {
            "feedback": feedback,
            "iteration": iteration,
            "current_phase": "done",
            "improvement_needed": False,
            "done": True
        }

    return {
        "feedback": feedback,
        "iteration": iteration,
        "improvement_needed": improvement_needed,
        "current_phase": "data"  # Go back to data processing with feedback
    }

def generate_final_report(state: SupervisorState) -> Dict[str, Any]:
    """Generate final competition report"""

    comp_info = state.get("competition_info", {})
    data_result = state.get("data_worker_result", {})
    trainer_result = state.get("trainer_result", {})
    validator_result = state.get("validator_result", {})
    submission_result = state.get("submission_result", {})

    prompt = f"""
Generate a final report for Kaggle competition: {state['competition_name']}

Competition Info:
- Type: {comp_info.get('problem_type', 'Unknown')}
- Metric: {comp_info.get('metric', 'Unknown')}
- Target: {comp_info.get('target_score', 'Unknown')}

Data Processing:
- Satisfy rate: {data_result.get('satisfy_rate', 0)}
- Iterations: {len(data_result.get('previous_results', []))}

Model:
- Type: {trainer_result.get('model_type', 'Unknown')}
- Scores: {trainer_result.get('scores', {})}

Validation:
- Passed: {validator_result.get('passed', False)}
- Recommendations: {validator_result.get('recommendations', [])}

Submission:
- Score: {submission_result.get('score', 'Unknown')}
- ID: {submission_result.get('submission_id', 'Unknown')}
- Analysis: {submission_result.get('analysis', {})}

Iterations completed: {state.get('iteration', 0)}

Create a comprehensive report with:
1. Executive summary
2. Methodology
3. Results analysis
4. Lessons learned
5. Recommendations for future improvements
"""

    response = llm.invoke([HumanMessage(content=prompt)]).content

    return {
        "final_report": response,
        "done": True
    }


def after_phase(state: SupervisorState) -> str:
    """Determine next phase"""
    if state.get("done", False):
        return "end"

    phase = state["current_phase"]

    # Правильный порядок фаз
    phase_flow = {
        "download": "analyze",
        "analyze": "data",
        "data": "train",
        "train": "validate",
        "validate": "submit",
        "submit": "analyze_result",
        "analyze_result": "feedback",
        "feedback": "generate_report" if state.get("iteration", 0) >= state.get("max_iterations", 5) else "data",
        "generate_report": "end"
    }

    next_phase = phase_flow.get(phase, "end")
    print(f"Phase transition: {phase} -> {next_phase}")
    return next_phase


def run_supervisor(competition_name: str, max_iterations: int = 5) -> Dict[str, Any]:
    """Run supervisor agent with Kaggle integration"""

    initial_state = {
        "competition_name": competition_name,
        "kaggle_manager": None,
        "competition_info": {},
        "session": None,
        "data_worker_result": {},
        "trainer_result": {},
        "validator_result": {},
        "submission_result": {},
        "current_phase": "download",
        "iteration": 0,
        "max_iterations": max_iterations,
        "feedback": "",
        "final_report": "",
        "last_score": None,
        "improvement_needed": False,
        "done": False
    }

    # Build graph
    graph = StateGraph(SupervisorState)
    graph.add_node("download", download_competition_data)
    graph.add_node("analyze", analyze_competition)
    graph.add_node("data", run_data_phase)
    graph.add_node("train", run_training_phase)
    graph.add_node("validate", run_validation_phase)
    graph.add_node("submit", submit_to_kaggle)
    graph.add_node("analyze_result", analyze_kaggle_result)
    graph.add_node("generate_report", generate_final_report)

    graph.set_entry_point("download")
    graph.add_edge("download", "analyze")
    graph.add_edge("analyze", "data")
    graph.add_edge("data", "train")
    graph.add_edge("train", "validate")
    graph.add_edge("validate", "submit")
    graph.add_edge("submit", "analyze_result")

    # Conditional edges from analyze_result
    def after_analyze(state: SupervisorState) -> str:
        if state.get("done", False):
            return "end"
        if state.get("improvement_needed", False) and state.get("iteration", 0) < state.get("max_iterations", 5):
            return "data"
        return "generate_report"

    graph.add_conditional_edges("analyze_result", after_analyze, {
        "data": "data",
        "generate_report": "generate_report",
        "end": END
    })
    graph.add_edge("generate_report", END)

    app = graph.compile()

    result = app.invoke(initial_state)

    print("=" * 60)
    print("KAGGLE COMPETITION FINAL REPORT")
    print("=" * 60)
    print(result.get("final_report", "No report generated"))

    if result.get("submission_result", {}).get("submission_id"):
        print(f"\n📊 Submission ID: {result['submission_result']['submission_id']}")
        print(f"📈 Score: {result.get('last_score', 'Unknown')}")

    return result