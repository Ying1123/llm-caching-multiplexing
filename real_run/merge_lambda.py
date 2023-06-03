"""
python3 merge_lambda.py --in result_facebook-opt-1.3b.json result_facebook-opt-13b.json
"""

import argparse
import json
from typing import Dict, Sequence, Optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True, nargs="+")
    parser.add_argument("--out-file", type=str, default="merged.json")
    args = parser.parse_args()

    contents = []
    for in_file in args.in_file:
        content = json.load(open(in_file, "r"))
        contents.extend(content)

    merged = {}

    for c in contents:
        i = c["id"]
        model_name = list(c["records"].keys())[0]
        if i in merged:
            merged[i]["records"][model_name] = c["records"][model_name]
        else:
            merged[i] = c

    values = list(merged.values())
    with open(args.out_file, "w") as fout:
        json.dump(values, fout, indent=2)
