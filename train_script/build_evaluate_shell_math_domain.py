#!/usr/bin/env python

import os, json

# dataset_names = ["train_only_public_api", "train_only_public_no_api", "train_only_tal_api", "train_only_tal_no_api", "train_tal_and_public_api", "train_tal_and_public_no_api"]
dataset_path = [
    {
        "dataset_name": "GSM8K",
        "dataset_path": "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/data/GSM8K/reformat_test.json",
    },
    {
        "dataset_name": "MATH",
        "dataset_path": "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/data/MATH/reformat_test.json",
    },
]

# model_paths = [
#     {
#         "model_name": "llama2-7B-math-domain-normal-cot-0128",
#         "model_paths": [
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-normal-cot-0128/checkpoint-60",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-normal-cot-0128/checkpoint-120",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-normal-cot-0128/checkpoint-180",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-normal-cot-0128/checkpoint-240",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-normal-cot-0128/checkpoint-300",
#         ],
#         "device_num": 1,
#     },
#     {
#         "model_name": "llama2-7B-math-domain-without-cot-0128",
#         "model_paths": [
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-without-cot-0128/checkpoint-60",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-without-cot-0128/checkpoint-120",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-without-cot-0128/checkpoint-180",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-without-cot-0128/checkpoint-240",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-without-cot-0128/checkpoint-300",
#         ],
#         "device_num": 1,
#     },
#     {
#         "model_name": "llama2-13B-math-domain-normal-cot-0128",
#         "model_paths": [
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-normal-cot-0128/checkpoint-60",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-normal-cot-0128/checkpoint-120",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-normal-cot-0128/checkpoint-180",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-normal-cot-0128/checkpoint-240",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-normal-cot-0128/checkpoint-300",
#         ],
#         "device_num": 1,
#     },
#     {
#         "model_name": "llama2-13B-math-domain-without-cot-0128",
#         "model_paths": [
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-without-cot-0128/checkpoint-60",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-without-cot-0128/checkpoint-120",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-without-cot-0128/checkpoint-180",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-without-cot-0128/checkpoint-240",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-without-cot-0128/checkpoint-300",
#         ],
#         "device_num": 1,
#     },
#     {
#         "model_name": "mistral-7B-math-domain-normal-cot-0128",
#         "model_paths": [
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-normal-cot-0128/checkpoint-60",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-normal-cot-0128/checkpoint-120",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-normal-cot-0128/checkpoint-180",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-normal-cot-0128/checkpoint-240",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-normal-cot-0128/checkpoint-300",
#         ],
#         "device_num": 1,
#     },
#     {
#         "model_name": "mistral-7B-math-domain-without-cot-0128",
#         "model_paths": [
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-without-cot-0128/checkpoint-60",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-without-cot-0128/checkpoint-120",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-without-cot-0128/checkpoint-180",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-without-cot-0128/checkpoint-240",
#             "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-without-cot-0128/checkpoint-300",
#         ],
#         "device_num": 1,
#     },
# ]


model_paths = [
    {
        "model_name": "llama2-7B-math-hcot-llm-mse-10-bs-128-0129",
        "model_paths": [
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-100",
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-7B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-200",
        ],
        "device_num": 1,
    },
    {
        "model_name": "llama2-13B-math-hcot-llm-mse-10-bs-128-0129",
        "model_paths": [
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-100",
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-100",
        ],
        "device_num": 1,
    },
    {
        "model_name": "mistral-7B-math-hcot-llm-mse-10-bs-128-0129",
        "model_paths": [
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-100",
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-200",
        ],
        "device_num": 1,
    },
]


output_dir = "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/prediction"
eval_script_save_path = (
    "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/eval_script"
)
log_folder = "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/eval_logs"

# Loop to generate and write shell scripts
for one_data_config in dataset_path:
    dataset_name, test_dataset_path = (
        one_data_config["dataset_name"],
        one_data_config["dataset_path"],
    )
    for one_model_config in model_paths:
        model_name, model_ckpts, device_num = (
            one_model_config["model_name"],
            one_model_config["model_paths"],
            one_model_config["device_num"],
        )
        log_dir_model_data = os.path.join(log_folder, model_name, dataset_name)
        os.makedirs(log_dir_model_data, exist_ok=True)
        if test_dataset_path.endswith(".jsonl"):
            with open(test_dataset_path, "r") as f:
                data_lines = f.readlines()
        else:
            with open(test_dataset_path, "r") as f:
                data_lines = json.load(f)

        data_lines_length = len(data_lines)

        num_process_models = (
            8 // device_num
        )  # one instance contains 8 gpus, this means how many tasks can be assigned in one instance

        # Calculate the number of tasks to assign to each process model
        task_count_per_model = data_lines_length // num_process_models
        remainder_tasks = data_lines_length % num_process_models

        # Create a list to store the start and end indices for each process model
        task_ranges = []
        start_idx = 0

        for i in range(num_process_models):
            end_idx = start_idx + task_count_per_model
            if i < remainder_tasks:
                end_idx += 1  # Distribute the remainder tasks equally among the first 'remainder_tasks' process models
            if i == num_process_models - 1:
                end_idx += 1  # add one more for last end idx
            task_ranges.append((start_idx, end_idx))
            start_idx = end_idx

        start_index_list = (
            "(" + " ".join([str(start) for start, end in task_ranges]) + ")"
        )
        end_index_list = "(" + " ".join([str(end) for start, end in task_ranges]) + ")"

        # create a list to store the gpu ids
        gpu_id_assignment = []

        if num_process_models == 8:
            gpu_id_assignment = [str(i) for i in range(8)]
        elif num_process_models == 4:
            for i in range(0, 8, 2):
                gpu_id_assignment.append(f"{i},{i+1}")
        else:
            # Handle other cases as needed
            pass

        row_gpu_ids = "("
        for one_id_tuple in gpu_id_assignment:
            row_gpu_ids += '"' + one_id_tuple + '"' + " "
        row_gpu_ids = row_gpu_ids.strip()
        row_gpu_ids += ")"

        output_model_data_dir = os.path.join(output_dir, model_name, dataset_name)
        eval_model_data_dir = os.path.join(
            eval_script_save_path, model_name, dataset_name
        )
        os.makedirs(output_model_data_dir, exist_ok=True)
        os.makedirs(eval_model_data_dir, exist_ok=True)

        for one_model_ckpt in model_ckpts:
            one_model_ckpt_basename = os.path.basename(one_model_ckpt)
            model_name_ckpt = model_name + "_{}".format(one_model_ckpt_basename)
            model_data_ckpt_num_dir = os.path.join(
                output_model_data_dir, one_model_ckpt_basename
            )
            os.makedirs(model_data_ckpt_num_dir, exist_ok=True)

            script_filename = os.path.join(
                eval_model_data_dir,
                "eval_{}.sh".format(
                    model_name
                    + "_"
                    + dataset_name
                    + "_"
                    + "{}".format(one_model_ckpt_basename)
                ),
            )

            # Write script to a shell script
            with open(script_filename, "w") as script_file:
                script_file.write("#!/bin/bash\n\n")
                script_file.write(
                    f"git config --global http.proxy http://10.202.1.3:18000\n"
                )
                script_file.write(
                    f"git config --global https.proxy http://10.202.1.3:18000\n"
                )
                script_file.write(
                    f"pip install wrapt_timeout_decorator sympy==1.11 jsonlines xlsxwriter -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                )
                if "mistral" in model_name:
                    script_file.write(
                        f"pip install transformers==4.34.0 -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                    )
                script_file.write(f'output_dir="{model_data_ckpt_num_dir}"\n')
                script_file.write(f"gpu_ids={row_gpu_ids}\n")
                script_file.write(f"model_path={one_model_ckpt}\n")
                script_file.write(f"model_name={model_name_ckpt}\n")
                script_file.write(f"data_name={dataset_name}\n")
                script_file.write(f"data_file={test_dataset_path}\n")
                script_file.write(f"device_num={device_num}\n")
                script_file.write(f"log_dir_model_data={log_dir_model_data}\n")
                script_file.write(f"start_index_list={start_index_list}\n")
                script_file.write(f"end_index_list={end_index_list}\n")
                script_file.write(f'mkdir -p "$output_dir"\n')
                script_file.write(
                    f"pids=() # Array to store the PIDs of the background processes\n\n"
                )

                script_file.write("for ((i=0; i<${#gpu_ids[@]}; i++)); do\n")
                script_file.write("    gpu_id=${gpu_ids[$i]}\n")
                script_file.write("    start_index=${start_index_list[$i]}\n")
                script_file.write("    end_index=${end_index_list[$i]}\n")

                script_file.write(
                    f'    echo "开始评估模型: $model_name, GPU ID: $gpu_id, Start index: $start_index, End index: $end_index"\n'
                )

                script_file.write(
                    f'    output_file="$output_dir/output_${{model_name}}_${{data_name}}_${{start_index}}-${{end_index}}_gpu-${{gpu_id}}.json"\n'
                )

                script_file.write(
                    f'    log_file="$log_dir_model_data/${{model_name}}_${{data_name}}_${{start_index}}-${{end_index}}_gpu-${{gpu_id}}.log"\n\n'
                )

                script_file.write(
                    '    command="CUDA_VISIBLE_DEVICES=$gpu_id nohup python -u /mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/evaluate_hcot_model_ed0.py'
                )  # 改inference的python文件
                if "HCOT" in model_name_ckpt:
                    script_file.write(f" --hcot_model")
                script_file.write(
                    f" --model_name $model_name --model_path $model_path --data_file $data_file"
                )
                script_file.write(
                    f' --output_file $output_file --start_index $start_index --end_index $end_index --gpu_id $gpu_id --device_num $device_num --max_new_tokens 512 > $log_file 2>&1 &"\n\n'
                )

                script_file.write('    eval "$command"\n')
                script_file.write(
                    "    pids+=($!) # Store the PID of the last command run in the background\n\n"
                )
                script_file.write("done\n")
                script_file.write('for pid in "${pids[@]}"; do\n')
                script_file.write('    wait "$pid"\n')
                script_file.write("done\n\n")

                script_file.write(
                    "# Kill all processes related to evaluating the model no api\n"
                )
                script_file.write("pkill -f evaluate_hcot_model_ed0.py\n\n")  # 改kill的名字

                script_file.write(
                    'echo "All evaluations completed and related processes killed."\n'
                )

            # Make the script executable
            os.chmod(script_filename, 0o755)
