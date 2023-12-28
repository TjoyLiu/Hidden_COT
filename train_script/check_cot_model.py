import os, json
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    BitsAndBytesConfig,
)
import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import gc


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def calculate(cal_json, logger=None):
    # TODO: should fix in future, ugly writing style
    try:
        api_name = cal_json["ActionName"].replace(" ", "")
        api_args = cal_json["Args"]
        result = api_map[api_name](api_args)
        output = {"result": result}
    except Exception as e:
        output = {"result": ""}
        if logger is not None:
            logger.error(f"Error in calculate: {e},input: {cal_json}")
    return output


def find_nearest_thought(generated_text):
    # Split the text at each occurrence of "<thought>"
    segments = generated_text.split("<thought>")

    # Return the last segment (which is the content after the last occurrence of "<thought>")
    return "<thought>" + segments[-1]


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
    current_prefix_with_api=5,
    bagging_times=5,
):
    print("Generating stream...")
    print(f"Params: {params}")
    use_plugin = params.get("use_plugin", True)
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)
    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    past_key_values = out = None
    gen_tokens = []
    for i in range(max_new_tokens):
        # print(f"Input ids: {tokenizer.decode(input_ids)}")
        if i == 0 or restart_gen:
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
            restart_gen = False
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        gen_tokens.append(token)
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                spaces_between_special_tokens=False,
            )
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                else:
                    raise ValueError("Invalid stop field type.")

            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


def load_json(file_path):
    print("file path here: ", file_path)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def load_jsonl(file_path):
    print("file path here: ", file_path)
    with open(file_path, "r") as file:
        raw_data = file.readlines()
    data = []
    for line in raw_data:
        data.append(json.loads(line))
    return data


def save_jsonl(file_path, data):
    # 写入JSONL文件
    with jsonlines.open(file_path, mode="w") as writer:
        writer.write_all(data)


def generate_chat_response_zero_shot(model, tokenizer, device, prompt, args):
    system_template = """{}"""
    params = {
        "prompt": system_template.format(prompt),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "stream_interval": 1,
    }
    completion = generate_stream(model, tokenizer, params, device)
    for one_text in completion:
        pass
    return one_text


def generate_chat_response_few_shot(model, tokenizer, device, question, previous, args):
    template = "Solve the following math word problem with your step-by-step analysis, {question}.I will provide the previous analysis and equations, my job is to show me your thought of the the next step calculation, I do not need to give the calculation equation.\n\n{previous}\n<[COT]>"
    params = {
        "prompt": template.format(question=question, previous=previous),
        "temperature": 0.1,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "use_plugin": True,
        "stream_interval": 1,
        "stop": "</[COT]>",
    }
    completion = generate_stream(model, tokenizer, params, device)
    for one_text in completion:
        pass
    return one_text


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # from model.modeling_llama import LlamaForSeqCausalLM as LlamaForCausalLM
    from model.modeling_llama import LlamaForCotCausalLM, LlamaForCotCausalLM

    device = torch.device("cuda:0")

    cot_model_path = "/mnt/pfs/zitao_team/liuchangkun1/repos/tiny-gpt/1215/output/7b-bs-128-mse-10-0/checkpoint-100"
    model_config = AutoConfig.from_pretrained(cot_model_path)
    model = LlamaForCotCausalLM.from_pretrained(
        cot_model_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cot_model_path)
    que = "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
    pre = "To solve this problem, we need to calculate the total cost Josh incurred and the final value of the house after repairs. Then we can find the profit by subtracting the total cost from the final value."
    template = f"Solve the following math word problem with your step-by-step analysis, {que}.I will provide the previous analysis and equations, my job is to show me your thought of the the next step calculation, I do not need to give the calculation equation.\n\n{pre}\n<[COT]>"
    params = {
        "prompt": template,
        "temperature": 0.1,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "use_plugin": True,
        "stream_interval": 1,
        "stop": "</[COT]>",
    }
    completion = generate_stream(model, tokenizer, params, device)
    for one_text in completion:
        print(one_text)
    print("final is: {}".format(one_text))
