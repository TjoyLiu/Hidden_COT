import subprocess
import os
import json
from multiprocessing import Pool

result_dir = "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/prediction"


def execute_a_py(model_name, data_name):
    command = f"python /mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/GPT4_evaluation/clean_extraction_file.py --model_name {model_name} --data_name {data_name} --result_dir {result_dir}"

    process = subprocess.Popen(command, shell=True)
    process.wait()


def main():
    with open(
        "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/GPT4_evaluation/need_run_extraction.json",
        "r",
    ) as json_file:
        data = json.load(json_file)

    batch_size = 20
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    for i, batch in enumerate(batches):
        print(f"Batch {i + 1}/{len(batches)}")
        with Pool(len(batch)) as pool:
            pool.starmap(execute_a_py, batch)


if __name__ == "__main__":
    main()
