"""Supervisor agent - orchestrates all other agents"""

import json
import pandas as pd
from typing import Dict, Any, TypedDict, List, Optional
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from utils import logger, SessionManager
from utils.benchmark import Benchmark
from config import get_config
from agents.data_worker import run_data_worker
from agents.trainer import run_trainer
from agents.validator import run_validator


class SupervisorState(TypedDict):
    competition_name: str
    session: Optional[SessionManager]
    data_processed: bool
    data_quality_score: float
    current_model_score: float
    best_model_score: float
    best_model_path: Optional[str]
    iterations: int
    max_iterations: int
    current_phase: str
    history: List[Dict[str, Any]]
    competition_info: Dict[str, Any]
    final_submission_made: bool
    submission_score: Optional[float]
    errors: List[str]


class SupervisorAgent:
    """Супервизор - оркестрирует работу всех агентов"""

    def __init__(self):
        self.config = get_config()
        self.llm = self.config.get_llm()
        self.benchmark = Benchmark(self.config.pipeline.benchmarks_dir)
        self._build_graph()

    def _build_graph(self):
        """Строит граф состояний"""
        graph = StateGraph(SupervisorState)

        graph.add_node("analyze_competition", self._analyze_competition)
        graph.add_node("process_data", self._process_data_phase)
        graph.add_node("train_model", self._train_model_phase)
        graph.add_node("validate_model", self._validate_model_phase)
        graph.add_node("analyze_results", self._analyze_results_phase)
        graph.add_node("improve_model", self._improve_model_phase)
        graph.add_node("make_submission", self._make_submission)

        graph.set_entry_point("analyze_competition")
        graph.add_edge("analyze_competition", "process_data")
        graph.add_conditional_edges("process_data", self._after_processing, {
            "train": "train_model",
            "retry": "process_data",
            "abort": END
        })
        graph.add_edge("train_model", "validate_model")
        graph.add_edge("validate_model", "analyze_results")
        graph.add_conditional_edges("analyze_results", self._after_analysis, {
            "improve": "improve_model",
            "submit": "make_submission",
            "abort": END
        })
        graph.add_edge("improve_model", "train_model")
        graph.add_edge("make_submission", END)

        self.app = graph.compile()

    def _analyze_competition(self, state: SupervisorState) -> Dict[str, Any]:
        """Анализирует соревнование и определяет стратегию"""
        logger.info("Analyzing competition...")

        # Загружаем информацию о соревновании
        competition_info_path = Path(self.config.pipeline.data_dir) / self.config.competition.files.competition_info
        competition_info = {}

        if competition_info_path.exists():
            with open(competition_info_path, 'r', encoding='utf-8') as f:
                content = f.read()
                competition_info = {
                    "description": content,
                    "problem_type": self.config.competition.problem_type,
                    "metric": self.config.competition.metric,
                    "target_column": self.config.competition.target_column
                }

        # Создаем сессию
        session = SessionManager(
            source_dir=self.config.pipeline.data_dir,
            session_dir=None
        )

        return {
            "competition_info": competition_info,
            "session": session,
            "data_processed": False,
            "data_quality_score": 0.0,
            "current_model_score": float('inf'),
            "best_model_score": float('inf'),
            "best_model_path": None,
            "iterations": 0,
            "max_iterations": state.get("max_iterations", self.config.pipeline.max_iterations),
            "current_phase": "analysis",
            "history": [],
            "final_submission_made": False,
            "submission_score": None,
            "errors": []
        }

    def _refinement_context_for_train(
        self, state: SupervisorState
    ) -> tuple[Optional[Dict[str, float]], str]:
        """Метрики прошлого обучения и JSON плана улучшения для следующего вызова кодера."""
        if state["iterations"] <= 0:
            return None, ""

        previous_scores: Optional[Dict[str, float]] = None
        for h in reversed(state["history"]):
            if h["phase"] == "training":
                previous_scores = (h.get("result") or {}).get("scores")
                break

        improvement_blob = ""
        for h in reversed(state["history"]):
            if h["phase"] == "improvement":
                imp = h.get("improvements") or {}
                improvement_blob = json.dumps(imp, indent=2, default=str)
                break

        return previous_scores, improvement_blob

    def _process_data_phase(self, state: SupervisorState) -> Dict[str, Any]:
        """Запускает обработку данных через data_worker"""
        logger.info("Starting data processing phase...")

        try:
            result = run_data_worker(
                session=state["session"],
                max_attempts=self.config.pipeline.max_attempts_per_agent
            )

            data_quality = result.get("satisfy_rate", 0.0)

            state["history"].append({
                "phase": "data_processing",
                "result": result,
                "quality_score": data_quality,
                "iteration": state["iterations"]
            })

            # Если качество данных низкое, пробуем улучшить
            if data_quality < 0.8 and state["iterations"] < state["max_iterations"]:
                return {
                    "data_processed": False,
                    "data_quality_score": data_quality,
                    "current_phase": "processing"
                }

            return {
                "data_processed": data_quality > 0.7,
                "data_quality_score": data_quality,
                "current_phase": "processing"
            }

        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            state["errors"].append(str(e))
            return {"data_processed": False, "data_quality_score": 0.0}

    def _train_model_phase(self, state: SupervisorState) -> Dict[str, Any]:
        """Запускает обучение модели через trainer"""
        logger.info(f"Starting training phase (iteration {state['iterations'] + 1}/{state['max_iterations']})...")

        try:
            prev_scores, improvement_ctx = self._refinement_context_for_train(state)
            result = run_trainer(
                session=state["session"],
                max_attempts=self.config.pipeline.max_attempts_per_agent,
                training_iteration=state["iterations"],
                improvement_context=improvement_ctx,
                previous_scores=prev_scores,
            )

            scores = result.get("scores", {})
            current_score = scores.get(self.config.competition.metric, float('inf'))

            # Обновляем лучший результат
            best_score = min(state["best_model_score"], current_score)
            best_model_path = None
            if current_score < state["best_model_score"]:
                best_model_path = str(state["session"].session_dir / "model.pkl")
                logger.info(f"New best model! Score: {current_score} (previous: {state['best_model_score']})")

            state["history"].append({
                "phase": "training",
                "result": result,
                "score": current_score,
                "iteration": state["iterations"]
            })

            # Сохраняем в бенчмарк
            self.benchmark.add_result(
                model_name=f"model_iter_{state['iterations']}",
                metrics=scores,
                metadata={"iteration": state["iterations"]}
            )

            return {
                "current_model_score": current_score,
                "best_model_score": best_score,
                "best_model_path": best_model_path,
                "current_phase": "training"
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            state["errors"].append(str(e))
            return {"current_model_score": float('inf')}

    def _validate_model_phase(self, state: SupervisorState) -> Dict[str, Any]:
        """Запускает валидацию модели через validator"""
        logger.info("Starting validation phase...")

        try:
            result = run_validator(
                session=state["session"],
                target_column=self.config.competition.target_column,
                metric=self.config.competition.metric,
                threshold=0.1
            )

            passed = result.get("passed", False)
            recommendations = result.get("recommendations", [])
            validation_score = result.get("scores", {}).get(self.config.competition.metric, float('inf'))

            state["history"].append({
                "phase": "validation",
                "result": result,
                "passed": passed,
                "validation_score": validation_score,
                "recommendations": recommendations,
                "iteration": state["iterations"]
            })

            return {
                "current_phase": "validation"
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            state["errors"].append(str(e))
            return {}

    def _analyze_results_phase(self, state: SupervisorState) -> Dict[str, Any]:
        """Анализирует результаты и решает, нужно ли улучшение"""
        logger.info("Analyzing results...")

        # Получаем последние результаты (без session объекта)
        last_training = None
        last_validation = None

        for h in reversed(state["history"]):
            if h["phase"] == "training" and last_training is None:
                # Создаем копию без session
                last_training = {k: v for k, v in h.items() if k != "session"}
                if "result" in last_training and "session" in last_training["result"]:
                    last_training["result"] = {k: v for k, v in last_training["result"].items() if k != "session"}
            if h["phase"] == "validation" and last_validation is None:
                last_validation = {k: v for k, v in h.items() if k != "session"}
                if "result" in last_validation and "session" in last_validation["result"]:
                    last_validation["result"] = {k: v for k, v in last_validation["result"].items() if k != "session"}

        current_score = last_training.get("score", float('inf')) if last_training else state["current_model_score"]
        validation_passed = last_validation.get("passed", False) if last_validation else False

        # Создаем безопасную версию истории для JSON
        safe_history = []
        for h in state["history"][-3:]:
            safe_h = {k: v for k, v in h.items() if k != "session"}
            if "result" in safe_h and "session" in safe_h["result"]:
                safe_h["result"] = {k: v for k, v in safe_h["result"].items() if k != "session"}
            safe_history.append(safe_h)

        # Формируем промпт для анализа
        analysis_prompt = f"""
        Analyze the model performance and decide next action:

        Competition: {state['competition_name']}
        Metric: {self.config.competition.metric} (lower is better)

        Current Results:
        - Model score: {current_score}
        - Best score so far: {state['best_model_score']}
        - Validation passed: {validation_passed}
        - Iteration: {state['iterations']}/{state['max_iterations']}

        History summary (last 3 steps):
        {json.dumps(safe_history, indent=2, default=str)}

        Decide:
        1. If score is good enough (< 1000 for MSE) -> submit
        2. If we have more iterations -> improve
        3. Otherwise -> submit current best

        Output JSON:
        {{
            "decision": "improve|submit|abort",
            "reason": "explanation",
            "expected_improvement": 0.0
        }}
        """

        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])

        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            analysis = json.loads(content)
            decision = analysis.get(
                "decision",
                "improve" if state["iterations"] < state["max_iterations"] - 1 else "submit",
            )
        except Exception as e:
            logger.warning(f"Failed to parse analysis response: {e}")
            decision = "improve" if state["iterations"] < state["max_iterations"] - 1 else "submit"

        state["history"].append({
            "phase": "analysis",
            "decision": decision,
            "iteration": state["iterations"],
            "current_score": current_score
        })

        logger.info(f"Analysis decision: {decision}")

        return {"current_phase": "analysis"}

    def _improve_model_phase(self, state: SupervisorState) -> Dict[str, Any]:
        """Планирует и применяет улучшения модели"""
        logger.info(f"Planning improvements (iteration {state['iterations'] + 1})...")

        # Получаем рекомендации из валидации
        last_validation = None
        for h in reversed(state["history"]):
            if h["phase"] == "validation":
                last_validation = h
                break

        recommendations = last_validation.get("recommendations", []) if last_validation else []

        # Получаем текущий код модели
        model_code_path = state["session"].session_dir / "training_code.py"
        current_code = ""
        if model_code_path.exists():
            with open(model_code_path, 'r') as f:
                current_code = f.read()

        # Сохраняем текущий код если его нет
        if not current_code and state["history"]:
            last_training = next((h for h in reversed(state["history"]) if h["phase"] == "training"), None)
            if last_training and "training_code" in last_training.get("result", {}):
                current_code = last_training["result"]["training_code"][:1000]

        # Формируем промпт для улучшения
        improvement_prompt = f"""
        Suggest specific code improvements for the next training iteration.

        Current model score: {state['current_model_score']}
        Best score: {state['best_model_score']}
        Validation feedback: {recommendations}

        Current training code (simplified):
        {current_code if current_code else "No previous code"}

        Suggest improvements focusing on:
        1. Feature engineering (handle categorical variables better)
        2. Hyperparameter tuning
        3. Different model architectures — prefer sklearn first: HistGradientBoostingRegressor/Classifier,
           RandomForest, ExtraTrees; only suggest xgboost/lightgbm if they are acceptable dependencies
        4. Handling missing values more effectively

        Output JSON:
        {{
            "improvements": ["specific improvement 1", "specific improvement 2"],
            "model_type": "hist_gradient_boosting|random_forest|xgboost|lightgbm",
            "hyperparameters": {{"param": "value"}},
            "feature_changes": ["add feature", "remove feature"]
        }}
        """

        response = self.llm.invoke([HumanMessage(content=improvement_prompt)])

        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            improvements = json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to parse improvements: {e}")
            improvements = {
                "improvements": [
                    "Use HistGradientBoostingRegressor with numeric-only features after encoding",
                    "Treat all non-numeric columns (object, string, category) before scaling",
                ],
                "model_type": "hist_gradient_boosting",
            }

        # Сохраняем план улучшений (без session объекта)
        try:
            state["session"].save_metadata(f"improvements_iter_{state['iterations']}", improvements)
        except:
            pass

        state["history"].append({
            "phase": "improvement",
            "improvements": improvements,
            "iteration": state["iterations"]
        })

        return {
            "iterations": state["iterations"] + 1,
            "current_phase": "improvement"
        }

    def _make_submission(self, state: SupervisorState) -> Dict[str, Any]:
        """Создаёт и отправляет файл для сабмишна"""
        logger.info("Creating submission file...")

        session_dir = state["session"].session_dir
        predictions_path = session_dir / "predictions.csv"

        if not predictions_path.exists():
            logger.error("No predictions found for submission")
            return {"final_submission_made": False}

        # Проверяем формат предсказаний
        predictions = pd.read_csv(predictions_path)

        # Создаём submission файл в правильном формате
        submission_path = session_dir / "submission.csv"

        # Формат для соревнования MWS
        if 'prediction' in predictions.columns:
            if 'index' in predictions.columns:
                submission = predictions[['index', 'prediction']]
            else:
                submission = pd.DataFrame({
                    'index': range(len(predictions)),
                    'prediction': predictions['prediction']
                })
        elif 'target' in predictions.columns:
            submission = pd.DataFrame({
                'index': range(len(predictions)),
                'target': predictions['target']
            })
        else:
            # Берём первую числовую колонку как предсказания
            numeric_cols = predictions.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                submission = pd.DataFrame({
                    'index': range(len(predictions)),
                    'prediction': predictions[numeric_cols[0]]
                })
            else:
                logger.error("Cannot determine prediction column format")
                return {"final_submission_made": False}

        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
        logger.info(f"Submission shape: {submission.shape}")
        logger.info(f"Submission head:\n{submission.head()}")

        # Пытаемся отправить в Kaggle
        submission_score = None
        try:
            from tools.kaggle_utils import submit_to_competition, check_submission_status

            logger.info(f"Submitting to Kaggle competition: {state['competition_name']}")
            submission_id = submit_to_competition(
                competition_name=state["competition_name"],
                submission_file=str(submission_path),
                message=f"Auto submission from agent system - Iteration {state['iterations']}"
            )

            if submission_id:
                logger.info(f"✅ Submission successful! ID: {submission_id}")

                # Проверяем статус и получаем score
                import time
                time.sleep(5)  # Ждём немного
                submissions = check_submission_status(state["competition_name"], limit=1)
                if submissions and len(submissions) > 0:
                    submission_score = submissions[0].get('public_score')
                    if submission_score:
                        logger.info(f"📊 Submission score: {submission_score}")
                        state["history"].append({
                            "phase": "submission",
                            "submission_id": submission_id,
                            "score": submission_score,
                            "iteration": state["iterations"]
                        })
            else:
                logger.warning("Submission to Kaggle failed, but local file saved")

        except Exception as e:
            logger.error(f"Kaggle submission error: {e}")
            logger.info("Local submission file saved for manual upload")

        return {
            "final_submission_made": True,
            "submission_score": submission_score,
            "current_phase": "submission"
        }

    def _after_processing(self, state: SupervisorState) -> str:
        """Определяет следующий шаг после обработки данных"""
        if state["data_processed"] or state["iterations"] >= state["max_iterations"]:
            return "train"
        elif state["iterations"] < state["max_iterations"]:
            return "retry"
        return "abort"

    def _after_analysis(self, state: SupervisorState) -> str:
        """Определяет следующий шаг после анализа результатов"""
        last_analysis = next((h for h in reversed(state["history"]) if h["phase"] == "analysis"), None)
        decision = last_analysis.get("decision", "submit") if last_analysis else "submit"

        # iterations поднимается в improve_model; последний допустимый train при iterations == max_iterations - 1
        if decision == "improve" and state["iterations"] < state["max_iterations"] - 1:
            logger.info(f"Decision: Improve model (iteration {state['iterations'] + 1})")
            return "improve"
        else:
            logger.info(f"Decision: Submit final model (score: {state['best_model_score']})")
            return "submit"

    def run(self, competition_name: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Запускает супервизора"""
        initial_state: SupervisorState = {
            "competition_name": competition_name,
            "session": None,
            "data_processed": False,
            "data_quality_score": 0.0,
            "current_model_score": float('inf'),
            "best_model_score": float('inf'),
            "best_model_path": None,
            "iterations": 0,
            "max_iterations": max_iterations,
            "current_phase": "analysis",
            "history": [],
            "competition_info": {},
            "final_submission_made": False,
            "submission_score": None,
            "errors": []
        }

        result = self.app.invoke(initial_state)

        # Выводим итоговую статистику
        print("\n" + "=" * 60)
        print("SUPERVISOR FINAL REPORT")
        print("=" * 60)
        print(f"Final phase: {result.get('current_phase')}")
        print(f"Best model score: {result.get('best_model_score', 'N/A')}")
        print(f"Data quality: {result.get('data_quality_score', 0):.2f}")
        print(f"Total iterations completed: {result.get('iterations', 0)}")
        print(f"Submission made: {result.get('final_submission_made', False)}")
        print(f"Submission score: {result.get('submission_score', 'N/A')}")

        # Показываем историю итераций
        print("\n" + "-" * 40)
        print("ITERATION HISTORY:")
        print("-" * 40)
        for h in result.get("history", []):
            if h["phase"] == "training":
                print(f"Iteration {h.get('iteration', '?')}: Score = {h.get('score', 'N/A')}")
            elif h["phase"] == "submission":
                print(f"Submission: ID={h.get('submission_id', 'N/A')}, Score={h.get('score', 'N/A')}")

        if result.get("errors"):
            print(f"\nErrors encountered: {len(result['errors'])}")
            for err in result["errors"][-3:]:
                print(f"  - {err[:100]}")

        return result


def run_supervisor(competition_name: str, max_iterations: int = 3) -> Dict[str, Any]:
    agent = SupervisorAgent()
    return agent.run(competition_name, max_iterations)