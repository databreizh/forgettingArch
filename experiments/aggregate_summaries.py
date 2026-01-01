import argparse
import json
import os
from glob import glob
from typing import List, Dict, Any

import csv


def _find_summary_files(input_dir: str, suffix: str) -> List[str]:
    """
    Cherche récursivement tous les fichiers *_summary.json (ou avec suffix)
    dans input_dir.
    """
    pattern = f"*{suffix}.json"
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for name in filenames:
            if name.endswith(suffix + ".json"):
                files.append(os.path.join(root, name))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Répertoire racine contenant les *_summary.json",
    )
    parser.add_argument(
        "--output",
        default="experiments/summary_all.csv",
        help="Fichier CSV de sortie",
    )
    parser.add_argument(
        "--suffix",
        default="_summary",
        help="Suffix des fichiers JSON (par défaut: _summary)",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_csv = args.output
    suffix = args.suffix

    if not os.path.isdir(input_dir):
        print(f"[ERROR] input_dir n'existe pas : {input_dir}")
        return

    print(f"[DEBUG] input_dir = {input_dir} (exists={os.path.isdir(input_dir)})")
    print(f"[DEBUG] searching for pattern: *{suffix}.json")

    json_files = _find_summary_files(input_dir, suffix)
    print(f"[DEBUG] found {len(json_files)} JSON files under {input_dir}")

    if not json_files:
        print("[INFO] No summaries to write.")
        return

    rows: List[Dict[str, Any]] = []
    all_keys = set()

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # On remonte 'metadata' à plat avec préfixe meta_
        metadata = data.pop("metadata", {}) or {}
        flat_meta = {f"meta_{k}": v for k, v in metadata.items()}

        row = {
            **data,
            **flat_meta,
            "source_file": path,
        }

        rows.append(row)
        all_keys.update(row.keys())

    # S'assurer que weighted_cost est bien dans les colonnes si présent dans les JSON
    # (si aucun JSON ne l'a, il ne sera pas là, ce qui est logique)
    fieldnames = sorted(all_keys)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] Wrote {len(rows)} summaries to {output_csv}")


if __name__ == "__main__":
    main()
