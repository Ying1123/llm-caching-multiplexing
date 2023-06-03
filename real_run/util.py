import time

import evaluate


def get_rouge(prediction, references):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=[prediction],
                            references=[references],
                            use_aggregator=False)
    print(results)
    return results


def construct_prompt_chat(sample):
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


def construct_prompt(query, output, task):
    # adopted from https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/prompt.jsonl
    if task == "sum":
        sys_prompt = "You are a helpful and precise assistant for checking the quality of the summarization of an article."
        user_prompt = f'''[Article]\n{query}
[The Start of the Summarization]\n{output}\n[The End of the Summarization]
[System]
We would like to request your feedback on the quality of the summarization in response to the article displayed above.\nPlease rate the relevance, accuracy, conciseness of the response. Please give an overall score on a scale of 1 to 10, where a higher score indicates higher overall quality.\nPlease first output a single line containing only one value indicating the score. In the subsequent line, please provide a comprehensive explanation of your evaluation.\n
'''
    elif task == "chat":
        sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
        user_prompt = f'''[Question]\n{query}
[The Start of the Answer]\n{output}\n[The End of the Answer]
[System]
We would like to request your feedback on the performance of the answer in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of the response. Please give an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only one value indicating the score. In the subsequent line, please provide a comprehensive explanation of your evaluation.\n
'''
    else:
        raise Exception("Unrecognized task")
    return [sys_prompt, user_prompt]


def extract_score(response):
    p = response.find("\n")
    score = response[:p]
    exp = response[p + 1:]
    return score, exp


def get_gpt4_eval(query, output, task):
    import openai
    MAX_GPT4_RETRY = 3

    prompt = construct_prompt(query, output, task)
    print("``` prompt to GPT-4 ```")
    print(prompt)

    ret = {"score": None, "explanation": None}
    for i in range(MAX_GPT4_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt[0]},
                    {"role": "user", "content": prompt[1]}],
                temperature=0.2,
                max_tokens=64,
            )
            content = response["choices"][0]["message"]["content"]
            ret["score"], ret["explanation"] = extract_score(content)
            print("``` eval result ```")
            print(ret)
            break
        except Exception as e:
            print(e)
            time.sleep(5)
    if ret["score"] is None:
        print(f"Failed after {MAX_GPT4_RETRY} retries.")
    return ret


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

