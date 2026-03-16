import json
from pathlib import Path
from typing import Dict, Optional

class BrailleTranslator:
    def __init__(self, map_path: str):
        path = Path(map_path)
        if not path.exists():
            raise FileNotFoundError(f"Translation map not found: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            self._map: Dict[str, str] = json.load(f)

    def translate(self, dots_string: str) -> Optional[str]:
        """
        Translates a string of dots (e.g. '134') to a character (e.g. 'm').
        Returns None if the dot combination is not in the dictionary (e.g., noise).
        """
        if not dots_string:
            return None
        return self._map.get(dots_string, None)