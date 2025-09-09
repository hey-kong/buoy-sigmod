import time

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import (
    StorageContext,
    QueryBundle,
    load_index_from_storage,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import Stemmer

from customed_statistic import global_statistic


class Retriever:
    def __init__(self, args):
        self.args = args

        # build vector retriever
        self.storage_context = StorageContext.from_defaults(persist_dir=args.docstore + "_vec")
        self.vec_index = load_index_from_storage(self.storage_context)
        self.vec_retriever = self.vec_index.as_retriever(similarity_top_k=args.similarity_top_k)

        self.docstore = SimpleDocumentStore.from_persist_path(args.docstore + "_docstore.pkl")
        # build bm25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.docstore,
            similarity_top_k=args.bm25_similarity_top_k,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )
        # build fusion retriever
        self.fusion_retriever = QueryFusionRetriever(
            [self.vec_retriever, self.bm25_retriever],
            similarity_top_k=args.similarity_top_k + args.bm25_similarity_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )

        # pruning strategy
        self.pruning_strategies = ['topk', 'dynamic']

    def bm25_retrieve(self, query_text):
        start = time.perf_counter()
        nodes = self.bm25_retriever.retrieve(query_text)
        end = time.perf_counter()
        global_statistic.add_to_list("bm25_retrieval_time", end - start)
        global_statistic.add_to_list("bm25_retrieved_nodes", len(nodes))

        if len(nodes) == 0:
            exit("No chunk retrieved")
        return nodes

    def vec_retrieve(self, query_text):
        start = time.perf_counter()
        query_bundle = QueryBundle(query_str=query_text)
        nodes = self.vec_retriever.retrieve(query_bundle)
        end = time.perf_counter()
        global_statistic.add_to_list("vec_retriever_time", end - start)
        global_statistic.add_to_list("vec_retrieved_nodes", len(nodes))

        if len(nodes) == 0:
            exit("No chunk retrieved")
        return nodes

    def fusion_retrieve(self, query_text):
        start = time.perf_counter()
        nodes = self.fusion_retriever.retrieve(query_text)
        nodes = nodes[:self.args.similarity_top_k]
        end = time.perf_counter()
        global_statistic.add_to_list("retrieval_time", end - start)
        global_statistic.add_to_list("retrieved_nodes", len(nodes))

        if len(nodes) == 0:
            exit("No chunk retrieved")
        return nodes
