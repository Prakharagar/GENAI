from datetime import datetime
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class PromptManager:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def register_prompt(self, name: str, template: str, description: str = "") -> str:
        pid = f"{name.replace(' ', '_')}_{int(time.time())}"
        payload = {
            "id": pid,
            "name": name,
            "template": template,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        path = self.prompts_dir / f"{pid}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info(f"Prompt registered: {pid}")
        return pid

    def load_prompt(self, prompt_id: str) -> Dict[str, Any]:
        p = self.prompts_dir / f"{prompt_id}.json"
        if not p.exists():
            raise FileNotFoundError(f"Prompt id {prompt_id} not found at {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def list_prompts(self) -> List[str]:
        return [p.stem for p in sorted(self.prompts_dir.glob("*.json"))]

