from functools import wraps
from typing import List, Dict, Callable
from dataclasses import dataclass, field
import threading
import json

# ---------- Decorator for Logging Method Calls ----------
def log_method_call(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        print(f"[LOG] Calling: {method.__name__} with args: {args} kwargs: {kwargs}")
        result = method(self, *args, **kwargs)
        print(f"[LOG] Result from {method.__name__}: {result}")
        return result
    return wrapper

# ---------- Prompt Version Data Class ----------
@dataclass
class PromptVersion:
    version_id: str
    prompt_text: str
    outputs: Dict[str, str] = field(default_factory=dict)  # input_id -> output
    scores: Dict[str, Dict[str, int]] = field(default_factory=dict)  # input_id -> {metric: score}

# ---------- Singleton Meta ----------
class SingletonMeta(type):
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

# ---------- Prompt Manager (Singleton) ----------
class PromptManager(metaclass=SingletonMeta):
    def __init__(self):
        self.versions: Dict[str, PromptVersion] = {}

    @log_method_call
    def add_prompt_version(self, version_id: str, prompt_text: str):
        self.versions[version_id] = PromptVersion(version_id, prompt_text)

    @log_method_call
    def generate_output(self, version_id: str, input_id: str, input_text: str, model_fn: Callable[[str], str]):
        version = self.versions[version_id]
        prompt = version.prompt_text + "\n\n" + input_text
        output = model_fn(prompt)
        version.outputs[input_id] = output

    @log_method_call
    def record_score(self, version_id: str, input_id: str, scores: Dict[str, int]):
        if version_id in self.versions:
            self.versions[version_id].scores[input_id] = scores

# ---------- Prompt Evaluator ----------
class PromptEvaluator:
    def __init__(self, criteria: List[str]):
        self.criteria = criteria

    @log_method_call
    def evaluate(self, output: str, reference: str) -> Dict[str, int]:
        # Dummy evaluation: compare lengths as proxy for scoring
        return {
            criterion: min(5, max(1, 5 - abs(len(output) - len(reference)) // 20))
            for criterion in self.criteria
        }

# ---------- Prompt Logger ----------
class PromptLogger:
    @staticmethod
    @log_method_call
    def save_to_json(data: Dict, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    @log_method_call
    def export_all_versions(manager: PromptManager, filepath: str):
        export_data = {
            version_id: {
                "prompt": version.prompt_text,
                "outputs": version.outputs,
                "scores": version.scores
            }
            for version_id, version in manager.versions.items()
        }
        PromptLogger.save_to_json(export_data, filepath)

# ---------- Mock LLM Function ----------
def mock_model(prompt: str) -> str:
    # Simulated LLM output
    return f"Simulated Output for:\n{prompt[:50]}..."

# ---------- Demo Run ----------
if __name__ == "__main__":
    # Step 1: Setup
    manager = PromptManager()
    evaluator = PromptEvaluator(criteria=["accuracy", "clarity", "relevance"])

    # Step 2: Add prompt versions
    manager.add_prompt_version("v1.0", "Summarize the following article in 3 sentences.")
    manager.add_prompt_version("v2.0", "You are an expert. Summarize the article in 3 key sentences.")

    # Step 3: Define input texts
    inputs = {
        "article1": "The EU announced a new policy to reduce carbon emissions by 55% by 2030. It includes regulations and taxes...",
        "article2": "Apple released a new iPhone model with better battery life and an improved camera system..."
    }

    # Step 4: Generate outputs
    for version_id in manager.versions:
        for input_id, text in inputs.items():
            manager.generate_output(version_id, input_id, text, mock_model)

    # Step 5: Evaluate outputs
    for version_id, version in manager.versions.items():
        for input_id, output in version.outputs.items():
            reference = "Placeholder reference summary."
            score = evaluator.evaluate(output, reference)
            manager.record_score(version_id, input_id, score)

    # Step 6: Export to JSON
    PromptLogger.export_all_versions(manager, "prompt_versions_output.json")
