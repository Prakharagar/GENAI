from pathlib import Path
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

class ExperimentLogger:
    def __init__(self, experiments_dir: Path):
        self.dir = experiments_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "index.jsonl"

    def log(self, entry: Dict[str, Any]) -> str:
        run_id = f"run_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        payload = {"run_id": run_id, "timestamp": datetime.now().isoformat(), **entry}
        run_path = self.dir / f"{run_id}.json"
        run_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with self.index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logging.info(f"Experiment logged: {run_id}")
        return run_id

