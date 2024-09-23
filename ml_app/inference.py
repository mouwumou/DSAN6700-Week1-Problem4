import json
import os

from .train import Model


def main():
    m = Model()
    data = os.getenv("DATA")
    if not data:
        raise ValueError("No data provided")

    data = json.loads(data)
    records = [
        {
            "dataset": m.dataset,
            "architecture": m.architecture,
            "features": m.eval,
            "data": record,
            "label": label,
        }
    for record, label in zip(data, m(data))]

    json.dump(records, open("out.json", "w"))