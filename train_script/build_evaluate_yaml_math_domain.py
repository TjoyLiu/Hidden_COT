#!/usr/bin/env python

import os


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
#         "task_name": "llama2-7B-math-cot-0128",
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
#         "task_name": "llama2-7B-math-nocot-0128",
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
#         "task_name": "llama2-13B-math-cot-0128",
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
#         "task_name": "llama2-13B-math-nocot-0128",
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
#         "task_name": "mistral-7B-math-cot-0128",
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
#         "task_name": "mistral-7B-math-nocot-0128",
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
        "task_name": "llama2-7B-math-hcot-0129",
    },
    {
        "model_name": "llama2-13B-math-hcot-llm-mse-10-bs-128-0129",
        "model_paths": [
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-100",
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/llama2-13B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-100",
        ],
        "device_num": 1,
        "task_name": "llama2-13B-math-hcot-0129",
    },
    {
        "model_name": "mistral-7B-math-hcot-llm-mse-10-bs-128-0129",
        "model_paths": [
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-100",
            "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/outputs/mistral-7B-math-hcot-llm-mse-10-bs-128-0129/checkpoint-200",
        ],
        "device_num": 1,
        "task_name": "mistral-7B-math-hcot-0129",
    },
]


output_dir = "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/prediction"
eval_script_save_path = (
    "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/eval_script"
)
log_folder = "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/eval_logs"

where_to_save_yaml = "/mnt/pfs/zitao_team/tianqiaoliu/run_CCE_tasks/hcot_eval/first_batch_math_domain_hcot"
if not os.path.exists(where_to_save_yaml):
    os.makedirs(where_to_save_yaml, exist_ok=True)

for one_data_config in dataset_path:
    dataset_name, test_dataset_path = (
        one_data_config["dataset_name"],
        one_data_config["dataset_path"],
    )
    for one_model_config in model_paths:
        model_name, model_ckpts, device_num, model_task_name = (
            one_model_config["model_name"],
            one_model_config["model_paths"],
            one_model_config["device_num"],
            one_model_config["task_name"],
        )
        eval_model_data_dir = os.path.join(
            eval_script_save_path, model_name, dataset_name
        )

        for one_model_ckpt in model_ckpts:
            one_model_ckpt_basename = os.path.basename(one_model_ckpt)
            model_name_ckpt = model_name + "_{}".format(one_model_ckpt_basename)
            ckpt_num = one_model_ckpt_basename.split("-")[1]
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
            yaml_file_name = os.path.join(
                where_to_save_yaml,
                "eval_{}_{}_{}.yaml".format(
                    model_name, dataset_name, one_model_ckpt_basename
                ),
            )
            shell_script_name = script_filename

            task_name = (
                "{}-{}-{}".format(model_task_name, dataset_name, ckpt_num)
                .replace("_", "-")
                .lower()
            )
            with open(yaml_file_name, "w") as yaml_file:
                yaml_file.write(f'apiVersion: "kubeflow.org/v1"\n')
                yaml_file.write(f"kind: PyTorchJob\n")
                yaml_file.write(f"metadata:\n")
                yaml_file.write(f"  name: {task_name}\n")
                yaml_file.write(f"  namespace: prod\n")
                yaml_file.write(f"spec:\n")
                yaml_file.write(f"  pytorchReplicaSpecs:\n")
                yaml_file.write(f"    Master:\n")
                yaml_file.write(f"      replicas: 1\n")
                yaml_file.write(f"      restartPolicy: Never\n")
                yaml_file.write(f"      template:\n")
                yaml_file.write(f"        metadata:\n")
                yaml_file.write(f"        spec:\n")
                yaml_file.write(f"          schedulerName: volcano\n")
                yaml_file.write(f"          containers:\n")
                yaml_file.write(f"            - name: pytorch\n")
                yaml_file.write(
                    f"              image: ccr-2gmah3kl-vpc.cnc.bj.baidubce.com/model/sft-qiao:v0.5\n"
                )
                yaml_file.write(f"              imagePullPolicy: IfNotPresent\n")
                yaml_file.write(f"              command:\n")
                yaml_file.write(f"                - bash\n")
                yaml_file.write(
                    f"                - {shell_script_name}\n"
                )  # 改shell脚本的路径
                yaml_file.write(f"              env:\n")
                yaml_file.write(f"                - name: NCCL_DEBUG\n")
                yaml_file.write(f"                  value: INFO\n")
                yaml_file.write(f"                - name: NCCL_IB_DISABLE\n")
                yaml_file.write(f'                  value: "0"\n')
                yaml_file.write(f"              resources:\n")
                yaml_file.write(f"                limits:\n")
                yaml_file.write(f"                  nvidia.com/gpu: 8\n")
                yaml_file.write(f"                  rdma/hca: 1\n")
                yaml_file.write(f"              securityContext:\n")
                yaml_file.write(f"                capabilities:\n")
                yaml_file.write(f'                  add: ["IPC_LOCK"]\n')
                yaml_file.write(f"              volumeMounts:\n")
                yaml_file.write(f"                - mountPath: /dev/shm\n")
                yaml_file.write(f"                  name: cache-volume\n")
                yaml_file.write(f"                - name: data\n")
                yaml_file.write(f"                  mountPath: /mnt/pfs\n")
                yaml_file.write(f"                - name: cfs-pvc\n")
                yaml_file.write(f"                  mountPath: /mnt/lck_cfs\n")
                yaml_file.write(f"          volumes:\n")
                yaml_file.write(f"            - name: cfs-pvc\n")
                yaml_file.write(f"              persistentVolumeClaim:\n")
                yaml_file.write(f"                claimName: pvc-cfs-ck\n")
                yaml_file.write(f"            - name: cache-volume\n")
                yaml_file.write(f"              emptyDir:\n")
                yaml_file.write(f"                medium: Memory\n")
                yaml_file.write(f"            - name: data\n")
                yaml_file.write(f"              persistentVolumeClaim:\n")
                yaml_file.write(f"                claimName: pfs-pvc-model\n")

            print(f"Generated YAML file: {yaml_file_name}")

print("YAML files generated for all combinations.")
