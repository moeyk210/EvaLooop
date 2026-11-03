"""Tests for the pseudocode loop, loop selection, Gemini config, and task subsetting."""

import pytest

from evaloop.core.config import GEMINI_OPENAI_BASE_URL, LOOP_TYPES, EvaluationConfig
from evaloop.core.evaluator import EvaLoopEvaluator
from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.models.base import BaseLLM
from evaloop.models.closed_source import APIGenerationError, OpenAILLM
from evaloop.tasks import CodePseudocodeTask, TaskFactory

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT


def make_config(tmp_path, **kwargs):
    return EvaluationConfig(model="gemini-2.0-flash", output_dir=str(tmp_path), **kwargs)


class TestCodePseudocodeTask:
    def test_registered(self):
        task = TaskFactory().create_task({"type": "code_pseudocode"})
        assert isinstance(task, CodePseudocodeTask)
        assert task.produces_testable_code is False

    def test_prompt_embeds_code_and_forbids_python(self):
        prompt = CodePseudocodeTask({}).create_prompt("def f(x): return x")
        assert "def f(x): return x" in prompt
        assert "pseudocode" in prompt.lower()
        assert "Do not use Python syntax" in prompt

    def test_extract_unwraps_single_fenced_block(self):
        raw = "```text\nFUNCTION f(x)\n    RETURN x\nEND FUNCTION\n```"
        assert CodePseudocodeTask({}).extract_output(raw) == "FUNCTION f(x)\n    RETURN x\nEND FUNCTION"

    def test_extract_keeps_unfenced_text(self):
        assert CodePseudocodeTask({}).extract_output("  FUNCTION f(x)\n RETURN x \n") == "FUNCTION f(x)\n RETURN x"

    def test_extract_neutralises_triple_quotes(self):
        assert '"""' not in CodePseudocodeTask({}).extract_output('RETURN """x"""')

    def test_postprocess_reattaches_assert_line(self):
        out = CodePseudocodeTask({}).postprocess(
            "FUNCTION similar_elements(a, b)",
            {"cycle_input": MBPP_STYLE_PROMPT, "task_id": "Mbpp/2", "cycle_number": 1},
        )
        assert out == '"""\nFUNCTION similar_elements(a, b)\n' + ASSERT_LINE + '\n"""\n'

    def test_assert_line_survives_multiple_cycles(self):
        task = CodePseudocodeTask({})
        text = MBPP_STYLE_PROMPT
        for cycle in range(1, 4):
            text = task.postprocess(f"PSEUDO {cycle}", {"cycle_input": text, "cycle_number": cycle})
        assert text.splitlines()[-2] == ASSERT_LINE


class EchoLLM(BaseLLM):
    def __init__(self):
        super().__init__({"name": "echo"})
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if prompt.startswith("Generate Python code"):
            return "```python\ndef similar_elements(a, b):\n    return set(a) & set(b)\n```"
        return "FUNCTION similar_elements(a, b)\n    RETURN intersection of a and b"


class TestLoopSelection:
    def test_default_loop_is_upstream_summarization(self, tmp_path):
        configs = make_config(tmp_path).get_task_configs()
        assert [c["type"] for c in configs] == ["code_generation", "code_summarization"]
        assert "write a python function to" in configs[1]["prompt_template"]

    def test_detailed_loop_matches_package_prompts(self, tmp_path):
        configs = make_config(tmp_path, loop="summarization_detailed").get_task_configs()
        assert configs[0]["prompt_template"] == (
            "Generate Python code for the following task:\n\n{description}\n\n"
            "Return only the code, without explanations or comments."
        )
        assert configs[1]["prompt_template"].startswith(
            "Summarize what the following code does in a detailed description:"
        )

    def test_pseudocode_loop_shares_generation_prompt(self, tmp_path):
        pseudo = make_config(tmp_path, loop="pseudocode").get_task_configs()
        detailed = make_config(tmp_path, loop="summarization_detailed").get_task_configs()
        assert pseudo[0] == detailed[0]
        assert pseudo[1]["type"] == "code_pseudocode"

    def test_named_variant_only_changes_pseudocode_prompt(self, tmp_path):
        named = make_config(tmp_path, loop="pseudocode_named").get_task_configs()
        pseudo = make_config(tmp_path, loop="pseudocode").get_task_configs()
        assert named[0] == pseudo[0]
        assert named[1]["type"] == "code_pseudocode"
        prompt = TaskFactory().create_task(named[1]).create_prompt("def f(x): return x")
        assert "Keep the function name and parameter names exactly" in prompt
        assert "built-in" not in prompt
        assert "def f(x): return x" in prompt

    def test_task_configs_are_copies(self, tmp_path):
        make_config(tmp_path, loop="pseudocode").get_task_configs()[0]["language"] = "rust"
        assert LOOP_TYPES["pseudocode"][0]["language"] == "python"

    def test_unknown_loop_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown loop"):
            make_config(tmp_path, loop="nope")

    def test_loop_in_experiment_name(self, tmp_path):
        assert "_pseudocode_" in make_config(tmp_path, loop="pseudocode").experiment_name

    def test_pseudocode_cycle_end_to_end(self, tmp_path):
        config = make_config(tmp_path, loop="pseudocode")
        factory = TaskFactory()
        tasks = [factory.create_task(c) for c in config.get_task_configs()]
        model = EchoLLM()
        result = BatchEvaluationCycle(model, tasks).run_single_cycle(
            [{"task_id": "Mbpp/2", "prompt": MBPP_STYLE_PROMPT}], 1
        )["cycle_results"][0]
        assert result["task_1_output"].startswith("def similar_elements")
        assert result["task_2_output"] == (
            '"""\nFUNCTION similar_elements(a, b)\n    RETURN intersection of a and b\n'
            + ASSERT_LINE + '\n"""\n'
        )
        assert "Write language-agnostic pseudocode" in model.prompts[1]


class TestGeminiConfig:
    def test_gemini_routes_to_openai_compatible_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        cfg = make_config(tmp_path).get_model_config()
        assert cfg["type"] == "openai"
        assert cfg["model_id"] == "gemini-2.0-flash"
        assert cfg["base_url"] == GEMINI_OPENAI_BASE_URL
        assert cfg["api_key"] == "g-key"
        assert cfg["max_tokens"] == 1024
        assert "request_params" not in cfg

    @pytest.mark.parametrize("model,effort", [
        ("gemini-2.5-flash-lite", "none"),
        ("gemini-3.5-flash-lite", "minimal"),
        ("gemini-3.1-flash-lite", "minimal"),
        ("gemini-3.5-flash", "minimal"),
    ])
    def test_thinking_disabled(self, tmp_path, model, effort):
        cfg = EvaluationConfig(model=model, output_dir=str(tmp_path)).get_model_config()
        assert cfg["request_params"] == {"reasoning_effort": effort}
        assert cfg["raise_on_error"] is True

    def test_api_failure_raises_when_configured(self, monkeypatch):
        llm = OpenAILLM({"name": "g", "api_key": "k", "model_id": "m", "raise_on_error": True})

        def boom(**kwargs):
            raise ValueError("network down")

        monkeypatch.setattr(llm.client.chat.completions, "create", boom)
        with pytest.raises(APIGenerationError):
            llm.generate("hi")

    def test_api_failure_returned_as_text_by_default(self, monkeypatch):
        llm = OpenAILLM({"name": "g", "api_key": "k", "model_id": "m"})
        monkeypatch.setattr(llm.client.chat.completions, "create",
                            lambda **kw: (_ for _ in ()).throw(ValueError("x")))
        assert llm.generate("hi").startswith("Error generating response")

    def test_backoff_grows_and_caps(self):
        llm = OpenAILLM({"name": "g", "api_key": "k", "model_id": "m"})
        assert [llm._backoff(n) for n in (1, 2, 3, 10)] == [3, 6, 12, 60]

    def test_request_params_sent_with_each_call(self, monkeypatch):
        llm = OpenAILLM({"name": "g", "api_key": "k", "model_id": "gemini-2.5-flash",
                         "request_params": {"reasoning_effort": "none"}})
        sent = {}

        class Msg:
            content = "ok"

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        def fake_create(**kwargs):
            sent.update(kwargs)
            return Resp()

        monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
        assert llm.generate("hi") == "ok"
        assert sent["reasoning_effort"] == "none"
        assert sent["temperature"] == 0.0

    def test_non_gemini_models_unchanged(self, tmp_path):
        cfg = EvaluationConfig(model="gpt-4", output_dir=str(tmp_path)).get_model_config()
        assert "base_url" not in cfg


class TestApiErrorFlagging:
    class PassAllTester:
        def test_code(self, code, task_id, entry_point):
            return {"success": True, "plus_details": []}

    def _run(self, tmp_path, cycle_results):
        evaluator = EvaLoopEvaluator(make_config(tmp_path, loop="pseudocode"))
        tasks = [TaskFactory().create_task(c) for c in evaluator.config.get_task_configs()]
        prompts = [{"task_id": r["task_id"], "prompt": "p"} for r in cycle_results]
        model_result = evaluator._initialize_model_result("m", prompts)
        problems = {r["task_id"]: {"entry_point": "f"} for r in cycle_results}
        next_prompts = evaluator._process_cycle_results(
            {"cycle_results": cycle_results}, model_result, tasks, problems, self.PassAllTester(), 1)
        return model_result, next_prompts

    def test_errored_loop_is_flagged(self, tmp_path):
        ok = {"task_id": "Mbpp/2", "task_1_output": "code", "task_2_output": "pseudo"}
        bad = {"task_id": "Mbpp/3", "task_1_output": "code", "task_2_error": "429"}
        model_result, next_prompts = self._run(tmp_path, [ok, bad])
        by_id = {r["task_id"]: r for r in model_result["prompt_results"]}
        assert "api_error" not in by_id["Mbpp/2"]
        assert by_id["Mbpp/3"]["api_error"] is True
        assert by_id["Mbpp/3"]["cycles"][0]["api_error"] is True
        assert [p["task_id"] for p in next_prompts] == ["Mbpp/2"]

    def test_run_aborts_when_every_prompt_errors(self, tmp_path):
        bad = [{"task_id": f"Mbpp/{i}", "task_1_error": "404 not found"} for i in (2, 3)]
        with pytest.raises(RuntimeError, match="404 not found"):
            self._run(tmp_path, bad)


class TestTaskSubset:
    IDS = [f"Mbpp/{i}" for i in range(2, 50)]

    def _select(self, tmp_path, **kwargs):
        return EvaLoopEvaluator(make_config(tmp_path, **kwargs))._select_task_ids(self.IDS)

    def test_all_tasks_by_default(self, tmp_path):
        assert self._select(tmp_path) == self.IDS

    def test_subset_is_reproducible_and_ordered(self, tmp_path):
        first = self._select(tmp_path, num_tasks=10, subset_seed=1)
        assert first == self._select(tmp_path, num_tasks=10, subset_seed=1)
        assert len(first) == 10
        assert first == [t for t in self.IDS if t in first]

    def test_seed_changes_subset(self, tmp_path):
        assert self._select(tmp_path, num_tasks=10, subset_seed=1) != self._select(tmp_path, num_tasks=10, subset_seed=2)

    def test_invalid_num_tasks(self, tmp_path):
        with pytest.raises(ValueError):
            make_config(tmp_path, num_tasks=0)
