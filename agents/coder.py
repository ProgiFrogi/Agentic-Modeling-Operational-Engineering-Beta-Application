"""Main model for code generation and execution"""

import re
from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from tools import extract_code, check_syntax, execute_with_saving
from utils import logger
from utils.guardrails import CodeSafetyChecker
from config import get_config
from agents.prompts import INITIAL_CODE_PROMPT, FIX_CODE_PROMPT
from rag import get_storage, ChunkType, results_to_text


class CodingAgentState(TypedDict):
    task: str
    current_code: str
    syntax_error: Optional[str]
    execution_error: Optional[str]
    execution_output: Optional[str]
    attempts: int
    max_attempts: int
    done: bool
    final_code: Optional[str]
    data_dir: Optional[str]
    extra_rules: str


class CoderAgent:
    """Агент для генерации и выполнения кода"""

    def __init__(self):
        self.config = get_config()
        self.rag = get_storage()
        self.llm = self.config.get_llm()
        self.code_safety = CodeSafetyChecker(self.config)
        self._build_graph()

    def _build_graph(self):
        """Строит граф состояний"""
        graph = StateGraph(CodingAgentState)

        graph.add_node("generate", self._generate_initial_code)
        graph.add_node("check_syntax", self._check_syntax_wrapper)
        graph.add_node("check_safety", self._check_safety_wrapper)
        graph.add_node("execute", self._execute_code)
        graph.add_node("fix", self._fix_code)

        graph.set_entry_point("generate")
        graph.add_edge("generate", "check_syntax")
        graph.add_conditional_edges("check_syntax", self._after_check, {
            "check_safety": "check_safety",
            "fix": "fix",
            "fail": END
        })
        graph.add_conditional_edges("check_safety", self._after_safety, {
            "execute": "execute",
            "fix": "fix",
            "fail": END
        })
        graph.add_conditional_edges("execute", self._after_execute, {
            "success": END,
            "fix": "fix",
            "fail": END
        })
        graph.add_edge("fix", "check_syntax")

        self.app = graph.compile()

    def _generate_initial_code(self, state: CodingAgentState) -> Dict[str, Any]:
        """Генерирует начальный код на основе задачи"""
        if not state['task'] or state.get("done", False):
            return {"done": True}

        rag_retrieval = self.rag.search_chunks(state['task'], chunk_type=ChunkType.CODE_SNIPPET,
                                               n_results=self.config.pipeline.rag_retrievals)
        rag_text = results_to_text(rag_retrieval, self.config.pipeline.rag_char_limit)
        logger.info(f"[Coder] Retrieved from rag to be inserted: {rag_text}")

        prompt = INITIAL_CODE_PROMPT.format(
            task=state['task'],
            references=rag_text,
            extra_rules=state.get('extra_rules', '')
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        code = extract_code(response.content)

        logger.info(f"[Coder] Generated code:\n{code}...")

        return {
            "current_code": code,
            "attempts": 1,
            "done": False,
            "syntax_error": None,
            "execution_error": None,
            "execution_output": None,
            "final_code": None
        }

    def _check_syntax_wrapper(self, state: CodingAgentState) -> Dict[str, Any]:
        """Проверяет синтаксис кода"""
        return check_syntax(state)

    def _check_safety_wrapper(self, state: CodingAgentState) -> Dict[str, Any]:
        """Проверяет безопасность кода"""
        code = state.get("current_code", "")
        is_safe, error = self.code_safety.check(code)

        if not is_safe:
            logger.warning(f"[Coder] Unsafe code detected: {error}")
            return {"syntax_error": f"Safety violation: {error}"}

        return {"syntax_error": None}

    def _execute_code(self, state: CodingAgentState) -> Dict[str, Any]:
        """Выполняет сгенерированный код"""
        code = state["current_code"]
        data_dir = state.get("data_dir")

        task_name = re.sub(r'[^\w\-_\. ]', '_', state['task'][:50])
        output_dir = f"execution_results/{task_name}"

        timeout = self.config.pipeline.execution_timeout
        success, output, result_dir = execute_with_saving(
            code, data_dir=data_dir, output_dir=output_dir, timeout=timeout
        )

        if success:
            logger.info(f"[Coder] Code executed successfully! Results in {result_dir}")
            return {
                "execution_output": output,
                "execution_error": None,
                "done": True,
                "final_code": code
            }
        else:
            logger.error(f"[Coder] Execution failed: {output}")
            return {
                "execution_error": output,
                "execution_output": None,
                "done": True
            }

    def _fix_code(self, state: CodingAgentState) -> Dict[str, Any]:
        """Исправляет код на основе ошибки"""
        error = state.get("syntax_error") or state.get("execution_error")

        prompt = FIX_CODE_PROMPT.format(
            code=state['current_code'],
            error=error
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        new_code = extract_code(response.content)

        logger.info(f"[Coder] Fixed code (attempt {state['attempts'] + 1})")

        return {
            "current_code": new_code,
            "attempts": state["attempts"] + 1,
            "syntax_error": None,
            "execution_error": None,
            "execution_output": None,
            "done": False
        }

    def _after_check(self, state: CodingAgentState) -> str:
        """Определяет следующий шаг после проверки синтаксиса"""
        if state["syntax_error"] and state["attempts"] < state["max_attempts"]:
            return "fix"
        elif not state["syntax_error"]:
            return "check_safety"
        return "fail"

    def _after_safety(self, state: CodingAgentState) -> str:
        """Определяет следующий шаг после проверки безопасности"""
        if state["syntax_error"]:
            if state["attempts"] < state["max_attempts"]:
                return "fix"
            return "fail"
        return "execute"

    def _after_execute(self, state: CodingAgentState) -> str:
        """Определяет следующий шаг после выполнения"""
        if state["execution_error"] is None:
            return "success"
        elif state["attempts"] < state["max_attempts"]:
            return "fix"
        else:
            return "fail"

    def run(self, task: str, max_attempts: int = 3, data_dir: Optional[str] = None, extra_rules: str = "") -> Dict[
        str, Any]:
        """Запускает агента-кодера"""
        initial_state: CodingAgentState = {
            "task": task,
            "current_code": "",
            "done": False,
            "syntax_error": None,
            "execution_error": None,
            "execution_output": None,
            "attempts": 0,
            "max_attempts": max_attempts,
            "final_code": None,
            "data_dir": data_dir,
            "extra_rules": extra_rules,
        }

        result = self.app.invoke(initial_state)

        print("=" * 50)
        print("Coder Results:")
        print(f"Success: {result.get('execution_error') is None}")
        print(f"Attempts: {result.get('attempts')}")
        if result.get('final_code'):
            print(f"Final code length: {len(result['final_code'])} chars")

        return result


# Для обратной совместимости
def run_coder(task: str, max_attempts: int = 3, data_dir: str | None = None, extra_rules: str = "") -> Dict[str, Any]:
    agent = CoderAgent()
    return agent.run(task, max_attempts, data_dir, extra_rules)
