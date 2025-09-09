import time

import torch
import torch.serialization
from transformers import AutoTokenizer, AutoModelForCausalLM

from customed_statistic import global_statistic

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


def get_model_type(model_name):
    model_lower = model_name.lower()
    if 'llama' in model_lower:
        return 'llama'
    elif 'mistral' in model_lower:
        return 'mistral'
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def query_prompt(chunk_list, query, model_name):
    model_type = get_model_type(model_name)
    prefix = PROMPT_PREFIX_MAP[model_type]
    suffix = PROMPT_SUFFIX_MAP[model_type]
    chunks = "\n\n".join(chunk_list)

    return (
        f"{prefix}{chunks}\n\n"
        f"Given the above context, answer the question: {query}\n\n"
        f"Only give me the answer and do not output any other words."
        f"{suffix}"
    )


class CustomModelWrapper:
    def init(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.model.config.eos_token_id
        self.model_type = get_model_type(model_path)

    def generate_answer(self, query, nodes):
        chunk_list = [node.text for node in nodes]
        prompt = query_prompt(chunk_list, query, self.model_type)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                do_sample=False
            )
        generated_ids = outputs[0]
        input_length = inputs.input_ids.shape[1]
        answer = self.tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True).strip()
        end = time.perf_counter()
        global_statistic.add_to_list("generate_time", end - start)
        return answer


llm = CustomModelWrapper()
