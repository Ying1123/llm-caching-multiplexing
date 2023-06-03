import argparse
import glob
import json
import os
import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from util import is_float


def extract_score(sentence):
    number = re.search("^\d+\.\d+", sentence)
    if number is None:
        number = re.search("^\d+", sentence)
    if number is None:
        return 0, 0
    score = float(number.group(0))
    if score < 0 or score > 10:
        print(sentence)
        print(score)
        raise Exception
    return score, 1


def ismodel(model_name, models):
    for model in models:
        if model in model_name:
            return True
    return False


def get_eval_files(args):
    files = []
    for filename in glob.glob(os.path.join(args.exp_dir, f"*_tagged.json")):
        model_name, query, _, _ = filename.split("/")[-1].split("_")
        num_query = query.split("-")[-1]
        if num_query != args.num_query: continue
        if ismodel(model_name, args.models):
            print(filename)
            files.append(filename)
    return files


def print_eval_results(eval_files):
    score = {}
    for filename in eval_files:
        model_name, query, _, _ = filename.split("/")[-1].split("_")
        if model_name not in score:
            score[model_name] = {}
            score[model_name]["mean"] = 0
            score[model_name]["num>=6"] = 0
            score[model_name]["cnt"] = 0
        with open(filename, "r") as f:
            data = json.load(f)

        for sample in data["samples"]:
            if not is_float(sample["gpt4_eval"]["score"]):
                continue
            value = float(sample["gpt4_eval"]["score"])
            score[model_name]["mean"] += value
            score[model_name]["cnt"] += 1
            if value >= 6:
                score[model_name]["num>=6"] += 1
        score[model_name]["mean"] /= score[model_name]["cnt"]
    print("scores:")
    for key in score:
        print(key, score[key])


def print_infer_time(eval_files):
    cost_min = [1e9, 1e9]
    cost_max = [0, 0]
    tcost = {}
    for i, filename in enumerate(eval_files):
        model_name, query, _, _ = filename.split("/")[-1].split("_")
        if model_name not in tcost:
            tcost[model_name] = 0
        with open(filename, "r") as f:
            data = json.load(f)
        for sample in data["samples"]:
            t = sample["infer_time"]
            tcost[model_name] += t
            cost_min[i] = min(cost_min[i], t)
            cost_max[i] = max(cost_max[i], t)
        tcost[model_name] /= len(data["samples"])
    print("mean inference time:")
    for key in tcost:
        print(key, tcost[key])
    print("min inference time:", cost_min)
    print("max inference time:", cost_max)


def print_seq_len(eval_files):
    for filename in eval_files:
        seq_len = []
        gen_len = []
        gen_time = []
        model_name, query, _, _ = filename.split("/")[-1].split("_")
        if "t5" in model_name:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
        elif "vicuna" in model_name:
            tokenizer = AutoTokenizer.from_pretrained("/home/Ying/models/vicuna-13b-v1.1", use_fast=False)
        else:
            raise Exception("unrecognized model")

        with open(filename, "r") as f:
            data = json.load(f)
            for sample in data["samples"]:
                input_ids = tokenizer(sample["query"]).input_ids
                seq_len.append(len(input_ids))
                output_ids = tokenizer(sample["output"]).input_ids
                gen_len.append(len(output_ids))
                gen_time.append(sample["infer_time"])
        print(filename)
        # print("seq len:", seq_len)
        # print("sum:", sum(seq_len))
        # print("gen len:", gen_len)
        # print("sum:", sum(gen_len))
        # print("time:", [f"{x:.1f}" for x in gen_time])
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="exp/oasst/a100")
    parser.add_argument("--models", nargs="+", default=["fastchat-t5", "vicuna"])
    parser.add_argument("--num-query", type=str, required=True)
    args = parser.parse_args()

    files = get_eval_files(args)
    print_eval_results(files)
    print_infer_time(files)
    print_seq_len(files)

