import os, json


def change_to_it_gsm8k(file_path):
    with open(file_path, "r") as f:
        file_content = f.readlines()
    transform_content = []
    for one_file_line in file_content:
        one_line_js = json.loads(one_file_line)
        question, answer = one_line_js["question"], one_line_js["answer"]
        analysis, answer = answer.split("\n####")
        analysis = analysis.strip("\n")
        answer = answer.strip("\n")
        one_output_line = {
            "analysis": analysis,
            "answer": answer,
            "instruction": "",
            "input": question,
        }
        transform_content.append(one_output_line)
    return transform_content


reformat_gsm8k = change_to_it_gsm8k(
    "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/data/GSM8K/test.jsonl"
)
with open(
    "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/data/GSM8K/reformat_test.json",
    "w",
) as f:
    json.dump(reformat_gsm8k, f, ensure_ascii=False)
