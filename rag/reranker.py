import time
import heapq
from FlagEmbedding import FlagReranker

from customed_statistic import global_statistic


class RerankerWrapper:
    def __init__(self):
        self.args = None
        self.reranker = None

    def init(self, args):
        model_path = 'BAAI/bge-reranker-v2-m3'
        self.reranker = FlagReranker(
            model_path,
            use_fp16=True,
            devices=["cuda:0"]
        )
        self.args = args
        print(f'use local reranker: {model_path}')

    def rerank_nodes(self, query_text, nodes, top_k=8):
        start = time.perf_counter()
        pairs = [(query_text, node.text) for node in nodes]

        scores = self.reranker.compute_score(pairs)

        topk = heapq.nlargest(top_k, zip(scores, nodes), key=lambda x: x[0])
        end = time.perf_counter()
        global_statistic.add_to_list("reranking_time", end - start)
        nodes = [node for _, node in topk]
        return sorted(nodes, key=lambda x: x.id_)


local_reranker = RerankerWrapper()
