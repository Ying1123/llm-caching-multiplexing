import argparse
import copy
import json
import os

from transformers import AutoTokenizer
from opt_config import get_opt_config


T = 1000 ** 4


def get_seq_len(tokenizer, sentence: str):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    return len(input_ids[0])


def flops_estimate(model_name, seq_len):
    config = get_opt_config(model_name)
    l = config.num_hidden_layers
    h1 = config.hidden_size
    h2 = config.ffn_embed_dim
    mm = l * (8 * seq_len * h1 ** 2 + 4 * seq_len * h1 * h2)
    bmm = l * (4 * seq_len ** 2 * h1)
    return (mm + bmm) / T


def main(args):
    data = []
    with open(args.file, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
    data_flops = []
    for sample in data:
        newsample = copy.deepcopy(sample)
        seq_len = get_seq_len(tokenizer, sample["sentence"])
        newsample["cost_small"] = flops_estimate(args.small_model, seq_len)
        newsample["cost_large"] = flops_estimate(args.large_model, seq_len)
        data_flops.append(newsample)

    assert len(data) == len(data_flops)

    with open(args.output, "w") as f:
        for sample in data_flops:
            json.dump(sample, f)
            f.write("\n")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../data/lambda_opt_clean_train.json")
    parser.add_argument("--output", type=str, default="../data/lambda_opt_flops_clean_train.json")
    parser.add_argument("--small-model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--large-model", type=str, default="facebook/opt-13b")
    args = parser.parse_args()

    main(args)
