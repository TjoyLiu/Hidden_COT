# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
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

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import random
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from conversation import SeparatorStyle, get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

try:
    from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

    replace_llama_attn_with_flash_attn()
except ImportError:
    print("Failed to import llama2_flash_attn_monkey_patch. Skip.")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="left", metadata={"help": "The padding side in tokenizer"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
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
    mse_ratio: float = field(default=2.0, metadata={"help": "MSE loss ratio"})
    addalign: bool = field(
        default=False,
        metadata={"help": "whether to add alignment loss in cot model training"},
    )
    addcontrastive: bool = field(
        default=False,
        metadata={"help": "whether to add alignment loss in cot model training"},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conv_template(
        "hcot"
    )  # 0411我更新了这个模版，这里变成了hcot不知道会不会有影响，因为之前都是none的，现在训练的cot model都是none的模版的结果。
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

    # only one turn is used for training
    vocab = tokenizer.get_vocab()
    cot_begin_token = vocab["<[COT]>"]
    cot_end_token = vocab["</[COT]>"]
    for target in targets:
        # find the last <[COT]> token
        cot_begin_idx = (target == cot_begin_token).nonzero(as_tuple=True)[0][-1]
        # find the first </[COT]> token
        cot_end_idx = (target == cot_end_token).nonzero(as_tuple=True)[0][0]
        assert (
            cot_begin_idx < cot_end_idx
        ), "COT begin token should be before COT end token"
        target[:cot_begin_idx] = IGNORE_TOKEN_ID
        # target[cot_end_idx:] = IGNORE_TOKEN_ID

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
        # print(self.raw_data[i])
        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # train_json = json.load(open(data_args.data_path, "r"))
    raw_data = load_all_data(data_args.data_path)
    train_dataset = dataset_cls(raw_data, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


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
    # for item in one_data:
    #     data_convert(item)
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


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        (
            model_args.model_name_or_path
            if model_args.model_name_or_path
            else model_args.config_name
        ),
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.mse_ratio = training_args.mse_ratio
    config.add_alignment = training_args.addalign
    config.add_contrastive = training_args.addcontrastive
    print("Add alignment loss: {}".format(config.add_alignment))
    print("Add contrastive loss: {}".format(config.add_contrastive))
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        (
            model_args.model_name_or_path
            if model_args.model_name_or_path
            else model_args.tokenizer_name_or_path
        ),
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    # Load model
    from model.modeling_llama import LlamaForCotCausalLM

    if model_args.model_name_or_path:
        model = LlamaForCotCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        model = LlamaForCotCausalLM(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )

    # If add special tokens, add this
    special_tokens_dict = {
        "additional_special_tokens": [
            "<[COT]>",
            "</[COT]>",
            "<equation>",
            "</equation>",
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    model.cot_begin_token_id = tokenizer.convert_tokens_to_ids("<[COT]>")
    model.cot_end_token_id = tokenizer.convert_tokens_to_ids("</[COT]>")

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
