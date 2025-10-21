import json
import os
import re
import string
from collections import Counter

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM

from buoy.transfer.quantization.cachegen_basics import CacheGenConfig
from buoy.transfer.quantization.kvcache_quant import quantize_dynamic_cache, dequantize_dynamic_cache
from buoy.transfer.kvcache_io import alloc_cpu_buffer, move_cache_to_cpu, move_cache_to_gpu, save_kvcache_quantized, load_kvcache_quantized

MODEL_PATH = "/data/llm/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
DTYPE = torch.float16

PROMPT_PREFIX_MAP = {
    'llama': (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
    ),
    'mistral': (
        "<s>[INST] "
    ),
}

PROMPT_SUFFIX_MAP = {
    'llama': (
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    ),
    'mistral': (
        " [/INST]"
    ),
}

root_dir = "longbench"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE).to(DEVICE)


def query_prompt(context, query):
    return (
        f"{context}\n\n"
        f"Given the above passages, answer the question: {query}\n\n"
        f"Only give me the answer and do not output any other words."
    )


# normalize_answer and f1_score are adapted from the official LongBench implementation:
# THUDM/LongBench (https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py)
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


for dataset in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset)
    questions_path = os.path.join(dataset_path, "questions", "questions.jsonl")
    documents_path = os.path.join(dataset_path, "documents")
    answers_path = os.path.join(dataset_path, "answers", "answers.jsonl")

    if not os.path.exists(questions_path) or not os.path.exists(documents_path) or not os.path.exists(answers_path):
        continue

    doc_files = sorted(
        [f for f in os.listdir(documents_path) if f.endswith(".txt")],
        key=lambda x: int(re.search(r"\d+", x).group())
    )

    answers = {}
    with open(answers_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            answers[data["id"]] = [normalize_answer(a) for a in data["answers"]]

    total_f1_prefix, total_f1_quant, count = 0.0, 0.0, 0
    with open(questions_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            q = json.loads(line)
            qid = q["id"]
            query = q["query"]

            if i < len(doc_files):
                doc_file = doc_files[i]
                with open(os.path.join(documents_path, doc_file), "r", encoding="utf-8") as df:
                    context = df.read().strip()
            else:
                context = ""

            prompt = query_prompt(context, query)

            if "llama" in MODEL_PATH.lower():
                model_key = "llama"
            elif "mistral" in MODEL_PATH.lower():
                model_key = "mistral"
            else:
                raise ValueError(f"Unknown model type from MODEL_PATH: {MODEL_PATH}")

            prefix = PROMPT_PREFIX_MAP[model_key]
            suffix = PROMPT_SUFFIX_MAP[model_key]

            inputs = tokenizer(prefix + prompt, return_tensors="pt").to(DEVICE)
            cfg = CacheGenConfig.from_model_name(MODEL_PATH)

            # 1) 得到原生 prefix_cache
            prefix_cache = DynamicCache()
            with torch.no_grad():
                prefix_cache = model(**inputs, past_key_values=prefix_cache, use_cache=True).past_key_values

            # 2) 量化保存
            kvcache_file_path = "./quant_cache.safetensors"

            # Step 1: 提取（在CPU上保留张量）
            buffer = alloc_cpu_buffer(prefix_cache)
            move_cache_to_cpu(prefix_cache, buffer)

            # Step 2: 量化（CPU）
            pack = quantize_dynamic_cache(prefix_cache, cfg)

            # Step 3: 保存
            save_kvcache_quantized(pack, kvcache_file_path)

            # 3) 反量化 -> legacy -> DynamicCache

            # Step 1: 只读量化数据
            pack = load_kvcache_quantized(kvcache_file_path)

            # Step 2: 反量化（CPU）
            kv_layers = dequantize_dynamic_cache(pack)

            # Step 3: 还原 DynamicCache
            quant_cache = DynamicCache.from_legacy_cache(kv_layers)

            move_cache_to_gpu(prefix_cache, device=torch.device(DEVICE))
            move_cache_to_gpu(quant_cache, device=torch.device(DEVICE))

            # test: with cache
            new_inputs = tokenizer(prefix + prompt + suffix, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output1 = model.generate(**new_inputs, past_key_values=prefix_cache, do_sample=False, use_cache=True,
                                         pad_token_id=tokenizer.eos_token_id,
                                         max_new_tokens=50)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generated_ids = output1[0]
            input_length = new_inputs.input_ids.shape[1]
            ans1 = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()

            # test: with quant_cache
            new_inputs = tokenizer(prefix + prompt + suffix, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output2 = model.generate(**new_inputs, past_key_values=quant_cache, do_sample=False, use_cache=True,
                                         pad_token_id=tokenizer.eos_token_id,
                                         max_new_tokens=50)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generated_ids = output2[0]
            input_length = new_inputs.input_ids.shape[1]
            ans2 = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()

            # === 对比标准答案，计算 F1 ===
            if qid in answers:
                ref_tokens_list = answers[qid]

                # prefix F1
                f1_a = max(f1_score(normalize_answer(ans1), ref) for ref in ref_tokens_list)
                # quant F1
                f1_b = max(f1_score(normalize_answer(ans2), ref) for ref in ref_tokens_list)

                total_f1_prefix += f1_a
                total_f1_quant += f1_b
                count += 1

                print(f"QID {qid}: prefix={f1_a:.4f}, quant={f1_b:.4f}")

    avg_prefix = total_f1_prefix / count if count else 0.0
    avg_quant = total_f1_quant / count if count else 0.0
    print(f"Dataset {os.path.basename(dataset_path)} - Avg F1 prefix={avg_prefix:.4f}, quant={avg_quant:.4f} ({count})")
