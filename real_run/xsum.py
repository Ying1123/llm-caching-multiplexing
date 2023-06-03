# use Open Assistant data: https://huggingface.co/datasets/OpenAssistant/oasst1
# use Flan-T5 3B and 11B: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/flan-t5#overview

import argparse
import json
import numpy as np
import os
import time
import torch
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import OPTForCausalLM

from util import get_rouge, get_gpt4_eval


MAX_GPT4_RETRY = 3


def construct_few_shot_sum(sample1, sample2, sample):
    prompt = ""
    prompt += "### " + "document:\n" + sample1["document"] + "\n"
    prompt += "### " + "summary:\n" + sample1["summary"] + "\n"

    prompt += "### " + "document:\n" + sample2["document"] + "\n"
    prompt += "### " + "summary:\n" + sample2["summary"] + "\n"

    prompt += "### " + "document:\n" + sample["document"] + "\n"
    prompt += "### " + "summary:\n"
    return prompt


def extract_answer(output):
    p = output.find("###")
    if p > 0:
        return output[:p]
    return output


def get_prompts(n):
    ds = load_dataset("xsum")
    test = ds["test"]
    prompts = []
    for i in tqdm(range(n)):
        prompt = {}
        prompt["id"] = test[i]["id"]
        prompt["article"] = test[i]["document"]
        r1 = np.random.randint(0, len(test))
        r2 = np.random.randint(0, len(test))
        prompt["prompt"] = construct_few_shot_sum(test[r1], test[r2], test[i])
        prompt["gt_summary"] = test[i]["summary"]
        prompts.append(prompt)
    return prompts


def get_model_outputs(model_name, prompts_path):
    print(f"loading {model_name} ...")
    kwargs = {"torch_dtype": torch.float16}
    if "facebook/opt" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OPTForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs).cuda()
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs).cuda()
    else:
        raise Exception("Unsupported model")
    print(f"{model_name} loaded.")

    with open(prompts_path, "r") as f:
        prompts_json = json.load(f)
    prompts = prompts_json["prompts"]
    print("number of queries", len(prompts))

    inputs = []
    for prompt in prompts:
        input_ids = tokenizer(prompt["prompt"], return_tensors="pt").input_ids.cuda()
        if len(input_ids[0]) < 1500:
            inputs.append(input_ids)
    print(f"tokenization finished.")

    # warmup
    warmup_num = 10
    pbar = tqdm(total=warmup_num, desc="warmup")
    start = time.time()
    for input_ids in inputs[:warmup_num]:
        model.generate(input_ids, max_new_tokens=64)
        pbar.update(1)
        if time.time() - start > 4:
            break
    pbar.close()

    # inference
    outputs = {"model_name": model_name,
               "query_path": prompts_path,
               "samples": [],
              }
    pbar = tqdm(total=len(inputs), desc="Finished queries")
    start = time.time()
    for i, input_ids in enumerate(inputs):
        tic = time.time()
        output_ids = model.generate(input_ids, max_new_tokens=128)
        duration = time.time() - tic
        output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        output = extract_answer(output)
        sample = {"id": prompts[i]["id"],
                  "query": prompts[i]["prompt"],
                  "article": prompts[i]["article"],
                  "output": output,
                  "infer_time": duration,
                  "gt_summary": prompts[i]["gt_summary"]
                 }
        outputs["samples"].append(sample)
        pbar.update(1)
    pbar.close()
    print(f"Finish all inference in {time.time() - start:.2f} s.")

    return outputs


def get_eval_results(data_file):
    with open(data_file, "r") as f:
        data = json.load(f)
    model_name = data["model_name"]
    query_path = data["query_path"]
    samples = data["samples"]

    for sample in samples:
        sample["rouge"] = get_rouge(sample["output"], [sample["gt_summary"]])
        sample["gpt4_eval"] = get_gpt4_eval(sample["article"], sample["output"], "sum")
    return data
    

def main(args):
    if args.gen_query > 0:
        # get prompts
        prompts_file = os.path.join(args.exp_dir, f"prompts-{args.gen_query}.json")

        print(f"\n====== generating {args.gen_query} prompts ======")
        prompts = get_prompts(args.gen_query)
        prompts_json = {"prompts": prompts}
        with open(prompts_file, "w") as f:
            json.dump(prompts_json, f)
        print(f"prompts dumped to {prompts_file}")

    if args.model is not None:
        # get model outputs
        assert args.device is not None, "device is none."
        assert args.query_path is not None, "query path is none."
        args.exp_dir = os.path.join(args.exp_dir, args.device)
        os.makedirs(args.exp_dir, exist_ok=True)
        model_name = args.model.split("/")[-1]
        query_file = args.query_path.split("/")[-1].split(".")[0]
        output_file = os.path.join(args.exp_dir, f"{model_name}_{query_file}_output.json")

        print(f"\n====== run inference for {model_name} on {query_file} ======")
        outputs = get_model_outputs(args.model, args.query_path)
        with open(output_file, "w") as f:
            json.dump(outputs, f)
        print(f"model output dumped to {output_file}")

    if args.eval_path is not None:
        eval_file = ".".join(args.eval_path.split(".")[:-1]) + "_tagged.json"
        print(f"\n====== run gpt4 eval for {args.eval_path} ======")
        eval_res = get_eval_results(args.eval_path)
        with open(eval_file, "w") as f:
            json.dump(eval_res, f)
        print(f"eval resuts dumped to {eval_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="exp/xsum")
    
    # generate query
    parser.add_argument("--gen-query", type=int, default=0)

    # generate model output
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--query-path", type=str, default=None)

    # generate eval results
    parser.add_argument("--eval-path", type=str, default=None,
                        help="The output json file path to get evaluated. "
                             "The evaluated file will be named with adding 'tag' in the input file name.")
    args = parser.parse_args()

    if args.model is not None and "flan" in args.model:
        args.model = "google/" + args.model
    os.makedirs(args.exp_dir, exist_ok=True)

    tic = time.time()
    main(args)
    print(f"whole program finished in {time.time() - tic:.2f} s.")
