import pandas as pd
import json
from call_gpt import run_multi_send_chat_request, send_chat_request
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_name", type=str)
    args = parser.parse_args()
    debug = False
    n_jobs = 30
    output_dir = (
        "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/prediction"
    )
    data_to_fill_record = []
    for one_model_name in os.listdir(output_dir):
        model_dir = os.path.join(output_dir, one_model_name)
        for one_data_name in os.listdir(model_dir):
            this_model_data_save_dir = os.path.join(model_dir, one_data_name)
            data_to_fill_record.append((one_model_name, one_data_name, output_dir))
    with open(
        "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/GPT4_evaluation/need_run_extraction.json",
        "w",
    ) as f:
        json.dump(data_to_fill_record, f, ensure_ascii=False)
