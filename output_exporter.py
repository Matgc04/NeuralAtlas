import os
import json
from typing import Dict, List

from pathlib import Path


class OutputExporter:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def _build_structure(self) -> Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]]:
        structure = {}

        for root, _, files in os.walk(self.output_dir):
            # Get relative path and split it into components
            rel_path = os.path.relpath(root, self.output_dir)
            if rel_path == ".":  # Ignore the root directory
                continue

            parts = rel_path.split(os.sep)
            if (
                len(parts) != 4
            ):  # Only process method level (model/dataset/class/method)
                continue

            model, dataset, class_name, method = parts
            images = sorted([f for f in files if (f.endswith(".jpg") or f.endswith(".webp"))])

            # Build nested data structure
            model_dict = structure.setdefault(model, {})
            dataset_dict = model_dict.setdefault(dataset, {})
            class_dict = dataset_dict.setdefault(class_name, {})
            class_dict[method] = images

        return structure

    def export_to_json(self, output_file: str):
        structure = self._build_structure()
        with open(Path(self.output_dir) / Path(output_file), "w") as f:
            json.dump(structure, f, indent=4)

if __name__ == "__main__":
    BASE_PATH = "interpretability-viewer/public/"
    OUTPUT_PATH = BASE_PATH + "outputs"
    exporter = OutputExporter(OUTPUT_PATH)
    exporter.export_to_json("outputs_structure.json")