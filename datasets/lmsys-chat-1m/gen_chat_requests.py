import json
import random
from datasets import load_dataset


def load_and_filter_dataset():
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    filtered = dataset.filter(lambda x: (
            x["model"] == "gpt-4"
            and x["language"] == "English"
            and x["redacted"] is False
    ))
    simplified_ds = filtered.map(
        lambda ex: {
            "conversation_id": ex["conversation_id"],
            "conversation": ex["conversation"],
            "turn": ex["turn"]
        },
        remove_columns=filtered.column_names
    )
    return simplified_ds


def extract_user_inputs(ds):
    conversations = []
    for idx, item in enumerate(ds):
        user_inputs = [msg['content'] for msg in item['conversation'] if msg['role'] == 'user']
        conversations.append({
            "conv_id": idx,
            "user_inputs": user_inputs,
        })
    return conversations


def generate_request_jsonl(conversations, output_path="request.jsonl"):
    random.seed(42)
    queues = {conv['conv_id']: list(enumerate(conv['user_inputs'])) for conv in conversations}
    urn = [conv['conv_id'] for conv in conversations]
    left = {conv['conv_id']: len(conv['user_inputs']) for conv in conversations}
    pos = {conv['conv_id']: 0 for conv in conversations}
    request_id = 0
    requests = []

    while sum(left.values()) > 0:
        available = [cid for cid in urn if left[cid] > 0]
        cid = random.choice(available)
        turn_idx = pos[cid]
        user_input = queues[cid][turn_idx][1]
        requests.append({
            'request_id': request_id,
            'conv_id': cid,
            'turn_id': turn_idx,
            'user_input': user_input
        })
        request_id += 1
        pos[cid] += 1
        left[cid] -= 1
        if left[cid] > 0:
            urn.append(cid)

    with open(output_path, 'w', encoding='utf-8') as fout:
        for req in requests:
            fout.write(json.dumps(req, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    simplified_ds = load_and_filter_dataset()
    conversations = extract_user_inputs(simplified_ds)
    generate_request_jsonl(conversations, output_path="request.jsonl")
