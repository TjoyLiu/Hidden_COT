# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()


import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

# from fastchat.conversation import SeparatorStyle
# from fastchat.model.model_adapter import get_conversation_template
from conversation import *

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/mnt/lck_cfs/model/liuchangkun1/models/checkpoint/huggingface_llama/llama-2-70b-tp8-pp1-dp64/iter_0002000"
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="/mnt/pfs/zitao_team/tianqiaoliu/Project/llm_team/source/training/sft/train_scripts/train_config/train_only_xuexiji.json",
        metadata={"help": "Path to the training data."},
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    output_dir: str = field(default="models/llama_SFT_code_self_train")


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# def preprocess(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     conv = get_conv_template("hcot")
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())
#     # Tokenize conversations
#     input_ids = tokenizer(
#         conversations,
#         return_tensors="pt",
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         truncation=True,
#     ).input_ids
#     targets = input_ids.clone()

#     assert conv.sep_style == SeparatorStyle.NO_COLON_THREE

#     # Mask targets
#     sep = conv.sep + conv.roles[1] + ": "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep2)
#         cur_len = 1
#         target[:cur_len] = IGNORE_TOKEN_ID
#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep
#             round_len = len(tokenizer(rou).input_ids)
#             instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#             target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

#             cur_len += round_len
#         target[cur_len:] = IGNORE_TOKEN_ID

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_TOKEN_ID
#                 rank0_print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.ne(tokenizer.pad_token_id),
#     )


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conv_template("hcot")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Apply ignore_token_id to the text before "ASSISTANT:"
    assistant_token_ids = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    for i, target in enumerate(targets):
        for idx in range(target.size(0) - len(assistant_token_ids) + 1):
            if target[idx : idx + len(assistant_token_ids)].equal(
                torch.tensor(assistant_token_ids)
            ):
                # Set tokens before this sequence to IGNORE_TOKEN_ID
                targets[i, : idx + len(assistant_token_ids)] = IGNORE_TOKEN_ID
                break
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def sample_jsonl(path, sample_ratio):
    data = []
    with open(path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    random.shuffle(data)  # 随机打乱
    data = data[: int(len(data) * sample_ratio)]  # 取样
    return data


def data_convert(data):
    if "prompt" in data and "response" in data:
        conversation = [
            {"from": "human", "value": data["prompt"]},
            {"from": "gpt", "value": data["response"]},
        ]
        data["conversations"] = conversation
        return data
    else:
        return data


def load_one_data(one_data):
    path = one_data["path"]
    sample_ratio = float(one_data["sample_ratio"])
    if sample_ratio == 0:
        # skip this file
        return []
    filetype = path.split(".")[-1]
    if filetype == "json":
        one_data = json.load(open(path, "r"))
        random.shuffle(one_data)  # 随机打乱
        one_data = one_data[: int(len(one_data) * sample_ratio)]  # 顺序采样
    elif filetype == "jsonl":
        one_data = sample_jsonl(path, sample_ratio)
    for item in one_data:
        data_convert(item)
    print(f"{path} has {len(one_data)} data, sample ratio {sample_ratio}")
    return one_data


def load_all_data(config_path):
    data_sources = json.load(open(config_path, "r"))
    raw_data = []
    for one_data in data_sources:
        one_data = load_one_data(one_data)
        raw_data += one_data
    print("total data:", len(raw_data))
    random.seed(42)
    random.shuffle(raw_data)
    return raw_data


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = load_all_data(data_args.data_path)
    print("total data:", len(raw_data))

    random.seed(42)
    random.shuffle(raw_data)
    train_dataset = dataset_cls(raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset)


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    special_tokens_dict = {
        "additional_special_tokens": [
            "<[COT]>",
            "<thought>",
            "</thought>",
            "<equation>",
            "</equation>",
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
