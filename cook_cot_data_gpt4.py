import pandas as pd
import json
from call_gpt import (
    run_multi_send_chat_request,
    send_chat_request,
    send_chat_request_async,
)
from tqdm import tqdm
import os
import glob
import random
import multiprocessing as mp
from functools import partial
import requests
import uuid
import time
import hashlib
import json
import argparse

# 接口地址&账号信息
# service_url = "http://genie.vdyoo.net/paas-proxy/sse/send"
# access_key_id = "200006"
# access_key_secret = "af6cf9e26e9c45cfb94d54236a5180de"
# sign_base = "{}&X-Genie-Timestamp={}&X-Genie-Nonce={}&X-Genie-DeviceId={}"
# device_id = "123456789"


# 计算验签值
# 输入参数的类型：字符串


def load_prompts(path):
    with open(path, "r", encoding="utf8") as f:
        return "".join(f.readlines())


def build_user_query(question, analysis, answer):
    analysis = analysis.split("\n\n\n")[0]
    return "USER: Question: {}\n Analysis: {}\n Answer:{}".format(
        question, analysis, answer
    )


def find_no_extraction(processed_file, file_to_process):
    file_to_fill = []
    processed_questions = []
    for line in processed_file:
        row_content = json.loads(line)

        if not isinstance(row_content["meta"], dict):
            row_content_meta = json.loads(row_content["meta"])
        else:
            row_content_meta = row_content["meta"]

        if "conversations" not in row_content_meta:
            if not isinstance(row_content_meta["meta"], dict):
                row_content_meta = json.loads(row_content_meta["meta"])
            else:
                row_content_meta = row_content_meta["meta"]

        processed_questions.append(row_content_meta["conversations"][-1]["value"])

    for line in file_to_process:
        line_to_process = json.loads(line)
        if "conversations" in line_to_process:
            line_conv = line_to_process["conversations"][-1]["value"]
        else:
            if not isinstance(line_to_process["meta"], dict):
                row_content_meta = json.loads(line_to_process["meta"])
            else:
                row_content_meta = line_to_process["meta"]
            line_conv = row_content_meta["conversations"][-1]["value"]
        if line_conv not in processed_questions:
            file_to_fill.append(line)
    return file_to_fill


def process_files(prompt_dir, keep_examples, system_input_md_path):
    system_input = load_prompts(system_input_md_path)
    if len(keep_examples) == 0:
        keep_examples = [
            x for x in os.listdir(prompt_dir) if "example" in x and x[0] != "."
        ]

    example_files = []
    for fname in keep_examples:
        file_path = os.path.join(prompt_dir, fname)
        example_files.append(file_path)
        # api_list.extend(extract_actions_from_path(file_path))
    # example_files = [x for x in example_files if "examples0.md" in x]
    examples_input = []
    for example_file in example_files:
        with open(example_file, "r") as f:
            example_lines = f.readlines()
        for example_line in example_lines:
            if example_line.startswith("USER"):
                example_line_user = example_line.strip("\n").replace("USER: ", "")
                examples_input.append({"role": "user", "content": example_line_user})
            elif example_line.startswith("ASSISTANT"):
                example_line_user = example_line.strip("\n").replace("ASSISTANT: ", "")
                examples_input.append(
                    {"role": "assistant", "content": example_line_user}
                )

    return system_input, examples_input


prompt_dir = "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/prompt"
system_input_info = (
    "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/prompt/instruction.md"
)
system_input_global, examples_input_global = process_files(
    prompt_dir, [], system_input_info
)


def build_input_data(row):
    """
    根据给定的数据集行，构建输入数据。
    """
    # idx, row = row
    # query = row.to_json()
    # row_content = json.loads(row)
    row_content = row
    user_query = build_user_query(
        row_content["question"], row_content["analysis"], row_content["answer"]
    )
    # gpt4
    data = {
        "system": system_input_global,
        "examples": examples_input_global,
        "question": user_query,
        "temperature": 0,
        "frequency_penalty": 1,
        "presence_penalty": 1,
        "engine": "magic:1000080061:2c44720c4e509196c4c65f6863d6de5f",
        "max_tokens": 512,
        "max_retry": 20,
    }
    # # wenxin
    # data = {
    #     "message": query,
    # }
    time.sleep(0.3)
    return data


def process_row(row):
    try:
        input_data = build_input_data(row)
        response = send_chat_request_async(**input_data)
        if response is not None:
            response["meta"] = row
        # response = CallWenXin(input_data)
        return response
    except Exception as e:
        print(f"Error processing row {row}: {e}")
        return None


# 定义处理数据集的函数
def process_dataset(dataset, output_file, n_jobs):
    # n_processes = min(cpu_count(), len(dataset))

    n_processes = n_jobs

    pbar = tqdm(total=len(dataset), desc="Processing files", dynamic_ncols=True)

    def wrapped_callback(response):
        if response is not None:
            with open(output_file, "a") as f:
                json.dump(response, f, ensure_ascii=False)
                f.write("\n")  # 每个响应为一行
        pbar.update(1)

    with mp.Pool(n_processes) as pool:
        for response in pool.imap_unordered(partial(process_row), dataset):
            wrapped_callback(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str)
    parser.add_argument("--output_data_path", type=str)
    parser.add_argument("--data_name", type=str)
    args = parser.parse_args()
    debug = False
    n_jobs = 200
    input_train_data = "/mnt/pfs/zitao_team/tianqiaoliu/Project/papers/hidden_cot/data/GSM8K/train.jsonl"
    # input_test_data = "/mnt/pfs/zitao_team/tianqiaoliu/Project/papers/hidden_cot/data/GSM8K/test.jsonl"
    with open(args.input_data_path, "r") as f:
        train_data_list = f.readlines()
    process_train_data = []
    if args.data_name == "gsm8k":
        for line in train_data_list:
            line_js = json.loads(line)
            answer_string = line_js["answer"]
            answer, analysis = (
                answer_string.split("\n####")[1].strip(),
                answer_string.split("\n####")[0].strip(),
            )
            process_train_data.append(
                {
                    "question": line_js["question"],
                    "answer": answer,
                    "analysis": analysis,
                }
            )
    elif args.data_name.lower() == "math":
        for line in train_data_list:
            line_js = json.loads(line)
            answer_string = line_js["answer"]
            analysis = line_js["output"]
            process_train_data.append(
                {
                    "question": line_js["input"],
                    "answer": answer_string,
                    "analysis": analysis,
                }
            )
    else:
        pass

    # n_jobs = 10
    process_dataset(process_train_data, args.output_data_path, n_jobs)
