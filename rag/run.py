import os
import json
import time
import argparse

import torch
from tqdm import tqdm
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from retriever import Retriever
from customed_statistic import global_statistic
from inference import llm
from reranker import local_reranker


def check_args(args) -> bool:
    if not os.path.exists(args.query_file):
        print(f"Query file {args.query_file} not found.")
        return False
    if not os.path.exists(args.answer_file):
        print(f"Answer file {args.answer_file} not found.")
        return False
    if not os.path.exists(args.docstore + "_docstore.pkl"):
        print(f"Docstore file {args.docstore} not found.")
        return False
    if not os.path.exists(args.docstore + "_vec"):
        print(f"Vector store dir {args.docstore} not found.")
        return False
    return True


def print_cmd(parser, args):
    command_lines = ["python3 run.py"]

    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.dest == "help":
            continue

        option = max(action.option_strings, key=lambda x: len(x))
        value = getattr(args, action.dest)

        if isinstance(value, bool):
            value = str(value)
            if value == "True":
                command_lines.append(f"    {option}")
            continue

        command_lines.append(f"    {option} {value}")
    formatted_command = " \\\n".join(command_lines)
    print(f"Command:\n{formatted_command}")


def main():
    # Parse command-line arguments at global scope
    parser = argparse.ArgumentParser(description='RAG Benchmarking Script')
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-small-en-v1.5',
                        help='Embedding model name or path')
    parser.add_argument('--query_file', type=str, default='../LongBench/qasper/questions/questions.jsonl',
                        help='Path to the file containing queries')
    parser.add_argument('--num_questions', type=int, default=0, help='Number of questions to process, 0 means all')
    parser.add_argument('--no_generate', action='store_true', default=False, help='Close generate stage for test')
    parser.add_argument('--answer_file', type=str, default='../LongBench/qasper/answers/answers.jsonl',
                        help='Path to the file containing answers')
    # use local llm
    parser.add_argument('--model_path', type=str, default='/data/llm/Llama-3.1-8B-Instruct',
                        help='Path of llm model')
    # retriever related (Basic: vectorIndex)
    parser.add_argument('--docstore', type=str, default='../docs_store/qasper', help='Path of nodes')
    parser.add_argument('--similarity_top_k', type=int, default=20, help='Top N of vector retriver')
    parser.add_argument('--bm25_similarity_top_k', type=int, default=20, help='Top N of BM25 retriever')
    parser.add_argument('--rerank_top_k', type=int, default=4, help='Top k')
    # log related
    parser.add_argument('--detailed_logging', action='store_true', help='Whether to enable detailed logging')
    args = parser.parse_args()
    if not check_args(args):
        return
    print_cmd(parser, args)

    # prepare stage
    global_statistic.init(args)
    llm.init(args.model_path)
    local_reranker.init(args)
    print("Loading index...")
    # Set up embedding model and load index
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)
    Settings.llm = None
    start = time.perf_counter()
    retriever = Retriever(args)
    end = time.perf_counter()
    global_statistic.add("retriever_init_time", end - start)

    # running stage
    print("Running benchmark...")
    questions = []
    with open(args.query_file, 'r', encoding='utf-8') as file:
        for item in file:
            item = json.loads(item)
            questions.append(item)
    if 0 < args.num_questions < len(questions):
        questions = questions[:args.num_questions]
    global_statistic.add("num_questions", len(questions))

    for item in tqdm(questions):
        query = item["query"]

        start = time.perf_counter()
        # retrieve
        nodes = retriever.fusion_retrieve(query)
        # rerank
        nodes = local_reranker.rerank_nodes(query, nodes, args.rerank_top_k)

        if not args.no_generate:
            # generate
            llm.generate_answer(query, nodes)
            end = time.perf_counter()
            global_statistic.add_to_list("rag_time", end - start)
            torch.cuda.empty_cache()

    global_statistic.dump()


if __name__ == "__main__":
    main()
