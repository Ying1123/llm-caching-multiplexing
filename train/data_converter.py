import argparse
import json
import os


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_model_name(model):
    if model in "flan-t5-xxl":
        return "flan-t5-xxl"
    elif model in "fastchat-t5-3b-v1.0":
        return "fastchat-t5-3b-v1.0"
    elif model in "vicuna-13b-v1.1":
        return "vicuna-13b-v1.1"
    else:
        raise Exception("unrecognized model")


def is_correct(gpt4_eval):
    if not is_float(gpt4_eval["score"]):
        return "NA"
    return int(float(gpt4_eval["score"]) > 5.9)


def main(args):
    models = [get_model_name(args.model_pair[0]),
              get_model_name(args.model_pair[1])]
    filenames = [models[0] + f"_prompts-{args.num_query}_output_tagged.json",
                 models[1] + f"_prompts-{args.num_query}_output_tagged.json"]
    files = [None] * 2
    data = [None] * 2
    samples= [None] * 2
    for i in range(2):
        files[i] = os.path.join(args.exp_dir, filenames[i])
        with open(files[i], "r") as f:
            data[i] = json.load(f)
            assert args.model_pair[i] in data[i]["model_name"].split("/")[-1]
            samples[i] = data[i]["samples"]
    assert len(samples[0]) == len(samples[1])

    raw_data = []
    for i in range(len(samples[0])):
        prompt = samples[0][i]["query"]
        assert prompt == samples[1][i]["query"]
        answer = [None] * 2
        time = [None] * 2
        correct = [None] * 2
        for j in range(2):
            answer[j] = samples[j][i]["output"]
            time[j] = samples[j][i]["infer_time"]
            correct[j] = is_correct(samples[j][i]["gpt4_eval"])

        sample = {"id": i, "records": {}}
        for j in range(2):
            if correct[j] == "NA": break
            sample["records"][models[j]] = [
                prompt, answer[j], time[j], correct[j], None]
        if correct[j] == "NA": continue

        raw_data.append(sample)

    with open(args.outfile, "w") as f:
        json.dump(raw_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="../real_run/exp/oasst/a100")
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--model-pair", nargs=2, type=str, required=True,
                        help="small first, then large model name")
    parser.add_argument("--num-query", type=int, required=True)
    args = parser.parse_args()

    if args.outfile is None:
        device = args.exp_dir.split("/")[-1]
        args.outfile = f"../data/oasst_{device}_{args.model_pair[0]}_{args.model_pair[1]}.json"

    main(args)
