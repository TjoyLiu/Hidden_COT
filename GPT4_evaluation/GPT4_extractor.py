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


def load_prompts(path):
    with open(path, "r", encoding="utf8") as f:
        return "".join(f.readlines())


def build_user_query(question, analysis):
    analysis = analysis.split("\n\n\n")[0]
    return "USER:{}\n\n模型生成的解析: {}".format(question, analysis)


def find_no_extraction(processed_file, file_to_process):
    file_to_fill = []
    processed_questions = []
    file_done_fill = []
    for line in processed_file:
        row_content = json.loads(line)

        if not isinstance(row_content["meta"], dict):
            row_content_meta = json.loads(row_content["meta"])
        else:
            row_content_meta = row_content["meta"]

        if "response" in row_content:
            processed_questions.append(row_content_meta["input"])
            file_done_fill.append(row_content)

    for line in file_to_process:
        line_to_process = json.loads(line)
        line_input = line_to_process["input"]
        if line_input not in processed_questions:
            file_to_fill.append(line)
    return file_to_fill, file_done_fill


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
    example_files = [x for x in example_files if "examples0.md" in x]
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


prompt_dir = "/mnt/pfs/zitao_team/tianqiaoliu/mathEval_data_check/matheval-lck/prompt"
system_input_info = "/mnt/pfs/zitao_team/tianqiaoliu/mathEval_data_check/matheval-lck/prompt/instruction.md"
system_input_global, examples_input_global = process_files(
    prompt_dir, [], system_input_info
)


def build_input_data(row):
    """
    根据给定的数据集行，构建输入数据。
    """
    # idx, row = row
    # query = row.to_json()
    row_content = json.loads(row)

    user_query = build_user_query(
        row_content["input"],
        row_content["response"],
    )
    # gpt4
    data = {
        "system": system_input_global,
        "examples": examples_input_global,
        "question": user_query,
        "temperature": 0,
        "frequency_penalty": 1,
        "presence_penalty": 1,
        "engine": "magic:1000080133:5a692de393b437bf98c6c3f36d273499",
        "max_tokens": 256,
        "max_retry": 20,
    }
    # # wenxin
    # data = {
    #     "message": query,
    # }
    # time.sleep(0.3)
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
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--result_dir", type=str)
    args = parser.parse_args()
    debug = False
    n_jobs = 200
    output_dir = args.result_dir
    model_name, data_name = args.model_name, args.data_name

    # in data name I have checkpoint num
    this_model_data_save_dir = os.path.join(output_dir, model_name, data_name)
    for one_ckpt_folder in os.listdir(this_model_data_save_dir):
        ckpt_result_folder = os.path.join(this_model_data_save_dir, one_ckpt_folder)
        all_json_files = glob.glob(os.path.join(ckpt_result_folder, "*.json"))

        for one_json_file in all_json_files:
            with open(one_json_file, "r") as f:
                json_content = f.readlines()
            base_json_file_name = os.path.basename(one_json_file)
            extraction_output_file = os.path.join(
                ckpt_result_folder,
                base_json_file_name.replace(".json", "") + "_gpt4_extraction.jsonl",
            )
            if os.path.exists(extraction_output_file):
                with open(extraction_output_file, "r") as f:
                    extractioned_lines = f.readlines()
                lines_to_fill, lines_done_fill = find_no_extraction(
                    extractioned_lines, json_content
                )
                if len(lines_done_fill) != len(json_content):
                    # first write a new file
                    with open(extraction_output_file, "w") as f:
                        pass
                    for done_line in lines_done_fill:
                        with open(extraction_output_file, "a") as fnew:
                            json.dump(done_line, fnew, ensure_ascii=False)
                            fnew.write("\n")  # 每个响应为一行

                    process_dataset(
                        lines_to_fill,
                        extraction_output_file,
                        n_jobs,
                    )
                    # for one_json_content in tqdm(json_content):
                    #     response = process_row(one_json_content)
                    #     with open(extraction_output_file, "a") as f:
                    #         f.write(json.dumps(response, ensure_ascii=False)+"\n")
                    print(
                        "Done for model {} data {} partition {}".format(
                            model_name, data_name, base_json_file_name
                        )
                    )
            else:
                print(data_name)
                print(model_name)
                print(extraction_output_file)
                print("No record for this")

                process_dataset(json_content, extraction_output_file, n_jobs)
                print(
                    "Done for model {} data {} partition {}".format(
                        model_name, data_name, base_json_file_name
                    )
                )
        # print("This is Done")
