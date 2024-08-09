import csv
import json

def read_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def write_json_output(results, file_path):
    with open(file_path, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=2)