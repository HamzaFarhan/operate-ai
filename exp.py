import json
from pathlib import Path

paths = {"path": Path("hmmm.json")}
file = Path("hmmm.json")
file.write_text(json.dumps(paths))
