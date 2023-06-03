import argparse
import json
import random

import numpy as np

from data_converter import get_model_name


def data_clean(raw_data, task, small_model, large_model):
    if task == "next_word" or task == "1-round-chat":
        clean_data = []
        for sample in raw_data:
            record1 = sample["records"][small_model]
            record2 = sample["records"][large_model]
            assert record1[0] == record2[0]
            clean_data.append({"sentence": record1[0],
                               "label": record1[3],
                               "correct_small": record1[3],
                               "correct_large": record2[3],
                               "cost_small": record1[2],
                               "cost_large": record2[2],
                               "idx": sample["id"],
                              })
    else:
        raise Exception(f"unrecognized task {task}")
    return clean_data


def get_portion(samples, small, large):
    ret = []
    for sample in samples:
        if (sample["correct_small"] == small and
            sample["correct_large"] == large):
            ret.append(sample)
    return ret


def main(args):
    with open(args.raw_data_path, "r") as f:
        raw_data = json.load(f)

    if "lambda" in args.raw_data_path:
        clean_data = data_clean(raw_data, "next_word", "facebook/opt-1.3b", "facebook/opt-13b")
    elif "oasst" in args.raw_data_path:
        models = args.raw_data_path.split(".")[-2].split("_")[-2:]
        small_model = get_model_name(models[0])
        large_model = get_model_name(models[1])
        clean_data = data_clean(raw_data, "1-round-chat", small_model, large_model)
    else:
        raise Exception("unrecognized raw data format")
    sclc = get_portion(clean_data, 1, 1)
    sclw = get_portion(clean_data, 1, 0)
    swlc = get_portion(clean_data, 0, 1)
    swlw = get_portion(clean_data, 0, 0)

    print("== before ==")
    print(f"#total: {len(clean_data)}")
    print(f"small correct, large correct: {len(sclc)}")
    print(f"small correct, large wrong  : {len(sclw)}")
    print(f"small wrong  , large correct: {len(swlc)}")
    print(f"small wrong  , large wrong  : {len(swlw)}")

    np.random.seed(0)
    if args.adjust_small_correct:
        num_sample = len(clean_data)
        num_sample = min((len(sclc) + len(sclw)) / (args.adjust_small_correct),
                         (len(swlc) + len(swlw)) / (1 - args.adjust_small_correct))
        target_sclc = int(num_sample * args.adjust_small_correct * (len(sclc) / (len(sclc) + len(sclw))))
        target_sclw = int(num_sample * args.adjust_small_correct * (len(sclw) / (len(sclc) + len(sclw))))
        target_swlc = int(num_sample * args.adjust_small_correct * (len(swlc) / (len(swlc) + len(swlw))))
        target_swlw = int(num_sample * args.adjust_small_correct * (len(swlw) / (len(swlc) + len(swlw))))

        sclc = list(np.random.choice(sclc, target_sclc, replace=False))
        sclw = list(np.random.choice(sclw, target_sclw, replace=False))
        swlc = list(np.random.choice(swlc, target_swlc, replace=False))
        swlw = list(np.random.choice(swlw, target_swlw, replace=False))
        clean_data = sclc + sclw + swlc + swlw

        print("== after ==")
        print(f"#total: {len(clean_data)}")
        print(f"small correct, large correct: {len(sclc)}")
        print(f"small correct, large wrong  : {len(sclw)}")
        print(f"small wrong  , large correct: {len(swlc)}")
        print(f"small wrong  , large wrong  : {len(swlw)}")

    perm = np.random.permutation(len(clean_data))
    clean_data = [clean_data[i] for i in perm]
    num_sample = len(clean_data)

    train = clean_data[:int(num_sample * 0.9)]
    valid = clean_data[int(num_sample * 0.9):int(num_sample * 0.99)]
    test = clean_data[int(num_sample * 0.99):]

    with open(args.clean_data_path, "w") as f:
        for sample in clean_data:
            json.dump(sample, f)
            f.write("\n")
    print(f"cleaned data dumped to {args.clean_data_path}")

    train_data_path = args.clean_data_path[:-len(".json")] + "_train.json"
    valid_data_path = args.clean_data_path[:-len(".json")] + "_valid.json"
    test_data_path = args.clean_data_path[:-len(".json")] + "_test.json"
    for kind in ["train", "valid", "test"]:
        data_path = eval(kind + "_data_path")
        with open(data_path, "w") as f:
            for sample in eval(kind):
                json.dump(sample, f)
                f.write("\n")
        print(f"{kind} data dumped to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", type=str, default="../data/lambda_opt.json")
    parser.add_argument("--clean-data-path", type=str, default=None)
    parser.add_argument("--adjust-small-correct", type=float, required=True)
    args = parser.parse_args()

    if args.clean_data_path is None:
        args.clean_data_path = args.raw_data_path.split("/")[-1].split(".")[0] + "_clean.json"

    main(args)
