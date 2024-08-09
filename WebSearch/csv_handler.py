import csv
import json
from typing import List, Dict, Any

def read_csv(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def write_json_output(results: List[Dict[str, str]], file_path: str) -> None:
    with open(file_path, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=2)