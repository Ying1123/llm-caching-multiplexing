# use Open Assistant data: https://huggingface.co/datasets/OpenAssistant/oasst1
# use Flan-T5 3B and 11B: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/flan-t5#overview

import argparse
from datasets import load_dataset
import json
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration


MAX_GPT4_RETRY = 3


def construct_few_shot_sum(sample1, sample2, sample):
    prompt = ""
    prompt += "title: " + sample1["info"]["title"] + "\n"
    prompt += "article:\n" + sample1["info"]["article"] + "\n"
    prompt += "summary:\n" + sample1["summary"]["text"] + "\n\n"

    prompt += "title: " + sample2["info"]["title"] + "\n"
    prompt += "article:\n" + sample2["info"]["article"] + "\n"
    prompt += "summary:\n" + sample2["summary"]["text"] + "\n\n"

    prompt += "title: " + sample["info"]["title"] + "\n"
    prompt += "article:\n" + sample["info"]["article"] + "\n"
    prompt += "summary:\n"
    print(prompt)
    print(sample1["info"]["id"])
    print(sample1["info"]["id"])
    print(sample["info"]["id"])
    return prompt


def get_prompts(n):
    ds = load_dataset("openai/summarize_from_feedback", "axis")
    test = ds["test"]
    print(test[0].keys())
    print(test[0]["info"])
    print()
    print(test[0]["summary"])
    prompts = []
    for i in range(2, len(test)):
        prompt = {}
        prompt["id"] = test[i]["info"]["id"]
        prompt["prompt"] = construct_few_shot_sum(test[0], test[1], test[i])
        exit()
        prompt["gt_summary"] = test[i]["summary"]["text"]
        prompts.append(prompt)
    return prompts


def get_model_outputs(model_name, prompts_path):
    print(f"loading {model_name} ...")
    kwargs = {"torch_dtype": torch.float16}
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, low_cpu_mem_usage=True, **kwargs).cuda()
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    with open(prompts_path, "r") as f:
        prompts_json = json.load(f)
    prompts = prompts_json["prompts"]
    print("number of queries", len(prompts))

    inputs = []
    for prompt in prompts:
        inputs.append(tokenizer(prompt[1], return_tensors="pt").input_ids.cuda())
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


def construct_prompt(sample):
    prompt = "Below is an user prompt and a model answer. " \
             "Please rate on how much it is likely to be correct and appropriate. " \
             "Print the score in the first line, and which should be in the range of [0, 10]. " \
             "score 0 means the answer is non-sense, score 10 means the answer is perfect." \
             "In the following lines, please provides the reasons of giving such a score."
    prompt += "User prompt:\n"
    prompt += sample["query"] + "\n"
    prompt += "Model answer:\n"
    prompt += sample["output"] + "\n"
    return prompt


def get_eval_results(data_file):
    import openai
    with open(data_file, "r") as f:
        data = json.load(f)
    model_name = data["model_name"]
    query_path = data["query_path"]
    samples = data["samples"]
    print(samples[0])

    for sample in samples:
        prompt = construct_prompt(sample)
        print("``` prompt to GPT-4 ```")
        print(prompt)

        sample["gpt4eval"] = None
        for i in range(MAX_GPT4_RETRY):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=32,
                )
                content = response["choices"][0]["message"]["content"]
                sample["gpt4eval"] = content
                print("``` eval ```")
                print(content)
                break
            except Exception as e:
                print(e)
                time.sleep(5)
        if sample["gpt4eval"] is None:
            print(f"Failed after {MAX_GPT4_RETRY} retries.")
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
        # get flan-t5 outputs
        assert args.device is not None, "device is none."
        assert args.query_path is not None, "query path is none."
        if args.exp_dir == DEFAULT_EXP_DIR:
            args.exp_dir = os.path.join(args.exp_dir, args.device)
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
    parser.add_argument("--exp-dir", type=str, default="exp/openai_sum")
    
    # generate query
    parser.add_argument("--gen-query", type=int, default=0)

    # generate model output
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        choices=["flan-t5-small", "flan-t5-base", "flan-t5-large",
                                 "flan-t5-xl", "flan-t5-xxl"],
                        help="flan-t5-small: 80M "
                             "flan-t5-base: 250M "
                             "flan-t5-large: 780M "
                             "flan-t5-xl: 3B "
                             "flan-t5-xxl: 11B",
                       )
    parser.add_argument("--query-path", type=str, default=None)

    # generate eval results
    parser.add_argument("--eval-path", type=str, default=None,
                        help="The output json file path to get evaluated. "
                             "The evaluated file will be named with adding 'tag' in the input file name.")
    args = parser.parse_args()

    if args.model is not None:
        args.model = "google/" + args.model
    os.makedirs(args.exp_dir, exist_ok=True)

    tic = time.time()
    main(args)
    print(f"whole program finished in {time.time() - tic:.2f} s.")
