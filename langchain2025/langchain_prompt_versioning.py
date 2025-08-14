# Requirements:
# pip install langchain openai


from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_loader import *

from dataclasses import dataclass, field
from typing import Dict, List
import json
import os

# ---------- Prompt Version Data Structure ----------
@dataclass
class PromptVersion:
    version_id: str
    template: PromptTemplate
    outputs: Dict[str, str] = field(default_factory=dict)
    scores: Dict[str, Dict[str, int]] = field(default_factory=dict)

# ---------- LangChain Prompt Manager ----------
class LangChainPromptManager:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.versions: Dict[str, PromptVersion] = {}

    def add_prompt_version(self, version_id: str, prompt_template: PromptTemplate):
        self.versions[version_id] = PromptVersion(version_id, prompt_template)

    def generate_outputs(self, version_id: str, inputs: Dict[str, str]):
        chain = LLMChain(llm=self.llm, prompt=self.versions[version_id].template)
        for input_id, input_text in inputs.items():
            output = chain.run(article=input_text)
            self.versions[version_id].outputs[input_id] = output

    def evaluate_outputs(self, version_id: str, criteria: List[str], reference: str = ""):
        for input_id, output in self.versions[version_id].outputs.items():
            self.versions[version_id].scores[input_id] = {
                criterion: min(5, max(1, 5 - abs(len(output) - len(reference)) // 20))
                for criterion in criteria
            }

    def export_results(self, filepath: str):
        export_data = {
            version_id: {
                "prompt": version.template.template,
                "outputs": version.outputs,
                "scores": version.scores
            }
            for version_id, version in self.versions.items()
        }
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=4)

# ---------- Sample Usage ----------
if __name__ == "__main__":
    # Make sure your OpenAI API key is set in your environment
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace or use dotenv

    manager = LangChainPromptManager()

    # Define Prompt Versions
    prompt_v1 = PromptTemplate.from_template("Summarize the following article in 3 sentences:\n\n{article}")
    prompt_v2 = PromptTemplate.from_template("You are an expert. Summarize the following article in 3 key sentences:\n\n{article}")

    manager.add_prompt_version("v1.0", prompt_v1)
    manager.add_prompt_version("v2.0", prompt_v2)

    # Input articles
    articles = {
        "article1": "The EU announced a new policy to reduce carbon emissions by 55% by 2030. It includes regulations and taxes...",
        "article2": "Apple released a new iPhone model with better battery life and an improved camera system..."
    }

    # Generate and evaluate
    manager.generate_outputs("v1.0", articles)
    manager.generate_outputs("v2.0", articles)

    manager.evaluate_outputs("v1.0", ["accuracy", "clarity", "relevance"], reference="Sample reference summary.")
    manager.evaluate_outputs("v2.0", ["accuracy", "clarity", "relevance"], reference="Sample reference summary.")

    # Export results
    manager.export_results("langchain_prompt_results.json")


