from typing import TypedDict

from langgraph.graph import StateGraph, END

from RAG.Vectorstore import VectorStore
from RAG.generator import Generator


class RAGState(TypedDict):
    question: str
    domain: str
    context: list[dict]
    answer: str


class RAGGraph:

    def __init__(self):

        self.vector_store = VectorStore()
        self.generator = Generator()

        graph = StateGraph(RAGState)

        # Nodes
        graph.add_node(
            "retrieve",
            self.retrieve,
        )

        graph.add_node(
            "generate",
            self.generate,
        )

        graph.add_node(
            "fallback",
            self.fallback,
        )

        # Entry point
        graph.set_entry_point(
            "retrieve"
        )

        # Decide what happens after retrieval
        graph.add_conditional_edges(
            "retrieve",
            self.route_after_retrieval,
            {
                "generate": "generate",
                "fallback": "fallback",
            },
        )

        graph.add_edge(
            "generate",
            END,
        )

        graph.add_edge(
            "fallback",
            END,
        )

        self.app = graph.compile()

    def retrieve(
        self,
        state: RAGState,
    ):

        results = self.vector_store.search(
            state["question"],
            top_k=3,
            domain=state["domain"],
        )

        return {
            "context": results
        }

    def route_after_retrieval(
        self,
        state: RAGState,
    ):

        context = state.get(
            "context",
            []
        )

        if not context:
            return "fallback"

        return "generate"

    def generate(
        self,
        state: RAGState,
    ):

        answer = self.generator.generate(
            state["question"],
            state["context"],
        )

        return {
            "answer": answer
        }

    def fallback(
        self,
        state: RAGState,
    ):

        return {
            "answer":
            "I could not find this information on the website."
        }

    def ask(
        self,
        question: str,
        domain: str,
    ):

        result = self.app.invoke(
            {
                "question": question,
                "domain": domain,
                "context": [],
                "answer": "",
            }
        )

        return result