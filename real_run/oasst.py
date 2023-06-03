# Open Assistant data: https://huggingface.co/datasets/OpenAssistant/oasst1
# Flan-T5 3B and 11B: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/flan-t5#overview

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import torch
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

from util import construct_prompt_chat, get_gpt4_eval


def get_prompts(n):
    ds = load_dataset("OpenAssistant/oasst1")
    train = ds["train"]
    prompts = []
    for i in range(len(train)):
        if train[i]["parent_id"] == None:
            prompts.append((train[i]["message_id"], train[i]["text"]))
            if len(prompts) == n:
                break
    print("number of queries", len(prompts))
    assert len(prompts) == n, "insufficient queries in open assistant dataset"
    # print("first 4 queries", prompts[:4])
    return prompts


def decorate(prompt):
    # prompt = "Question: " + prompt + "\n\nAnswer:"
    return prompt

def get_model_outputs(model_name, prompts_path):
    print(f"loading {model_name} ...")
    kwargs = {"torch_dtype": torch.float16}
    if "gpt4" in model_name:
        pass
    elif "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs).cuda()
    elif "vicuna" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs).cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs).cuda()

    with open(prompts_path, "r") as f:
        prompts_json = json.load(f)
    prompts = prompts_json["prompts"]
    print("number of queries", len(prompts))

    inputs = []
    for prompt in prompts:
        inputs.append(tokenizer(decorate(prompt[1]), return_tensors="pt").input_ids.cuda())
    print(f"tokenization finished.")

    # warmup
    warmup_num = 10
    pbar = tqdm(total=warmup_num, desc="warmup")
    start = time.time()
    for input_ids in inputs[:warmup_num]:
        model.generate(input_ids, max_new_tokens=128)
        pbar.update(1)
        if time.time() - start > 4:
            break
    pbar.close()

    # inference
    outputs = {"model_name": model_name,
               "query_path": prompts_path,
               "samples": [],
              }
    pbar = tqdm(total=len(inputs), desc="Finished quries")
    start = time.time()
    for i, input_ids in enumerate(inputs):
        tic = time.time()
        output_ids = model.generate(input_ids, max_new_tokens=128)
        duration = time.time() - tic
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        sample = {"oasst_message_id": prompts[i][0],
                  "query": prompts[i][1],
                  "output": output,
                  "infer_time": duration,
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

    results = []
    args_list = [(sample["query"], sample["output"], "chat") for sample in samples]
    with ThreadPoolExecutor(10) as executor:
        futures = {executor.submit(get_gpt4_eval, *args) for args in args_list}
        for future in as_completed(futures):
            results.append(future.result())
    print(results)
    results = list(results)
    for i, sample in enumerate(samples):
        sample["gpt4_eval"] = results[i]
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


# "flan-t5-small: 80M "
# "flan-t5-base: 250M "
# "flan-t5-large: 780M "
# "flan-t5-xl: 3B "
# "flan-t5-xxl: 11B"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="exp/oasst")
    
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

    if args.model is not None and args.model.startswith("flan-t5"):
        args.model = "google/" + args.model

    os.makedirs(args.exp_dir, exist_ok=True)

    tic = time.time()
    main(args)
    print(f"whole program finished in {time.time() - tic:.2f} s.")
