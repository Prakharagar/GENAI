# config_loader.py
import yaml
from pathlib import Path
import os

class Config:
    _instance = None
    _config_data = None
    _file_path='D:\\projects'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(cls._file_path)
            cls._instance._set_env()
        return cls._instance

    def _load_config(self,path):
        config_path = Path(path) / "config.yaml"
        with open(config_path, "r") as f:
            self._config_data = yaml.safe_load(f)    

    def _set_env(self):
        for _k,_v in self._config_data.items():
            for _ck,_cv in _v.items():
                _val=f'{_k}_{_ck}'
                os.environ[_val] = str(_cv)

    def get(self, section: str, key: str = None):
        if key:
            return self._config_data.get(section, {}).get(key)
        return self._config_data.get(section)

# Global access
config = Config()

