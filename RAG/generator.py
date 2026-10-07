import ollama


class Generator:
    def __init__(self, model: str = "llama3"):
        self.model = model

    def generate(
        self,
        question: str,
        context_chunks: list[dict],
    ) -> str:

        context = "\n\n".join(
            chunk["text"]
            for chunk in context_chunks
        )

        prompt = f"""
You are a helpful assistant answering questions about a website.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find this information on the website."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]