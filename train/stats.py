import argparse
import json
import numpy as np
import os
import re


def print_basics(args):
    with open(args.data_path, "r") as f:
        data = f.readlines()
        samples = [json.loads(sample) for sample in data]

    print(f"Stats about {args.data_path}")

    num_sample = len(samples)
    print(f"num sample: {num_sample}")

    labels = [sample["label"] for sample in samples]
    print("num label-1 samples", sum(labels))


def print_predicts(args):
    test_data_path = args.data_path[:-len(".json")] + "_test.json"
    with open(test_data_path, "r") as f:
        test_data = f.readlines()
        test_samples = [json.loads(sample) for sample in test_data]

    gt_labels = np.array([sample["label"] for sample in test_samples])

    predicts_path = os.path.join(args.model_ckpt, "predict_results_None.txt")
    with open(predicts_path, "r") as f:
        lines = f.readlines()
        predicts = []
        for line in lines:
            value = re.findall(r'\d+', line)
            if len(value) > 0:
                predicts.append(int(value[1]))
    predicts = np.array(predicts)
    assert len(gt_labels) == len(predicts)
    print("num label-1 samples in test data", np.sum(gt_labels))
    print("num label-1 samples in predictions", np.sum(predicts))
    accuracy = np.sum(gt_labels == predicts) / len(test_samples)
    print(f"predict accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../data/lambda_opt_clean.json")
    parser.add_argument("--model-ckpt", type=str, default="models/choice_model")
    args = parser.parse_args()

    print_basics(args)
    if os.path.exists(args.model_ckpt):
        print_predicts(args)

