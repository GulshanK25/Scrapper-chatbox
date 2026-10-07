from RAG.embeddings import EmbeddingModel
import numpy as np


model = EmbeddingModel()

documents = [
    "Python functions are defined using the def keyword.",
    "I really enjoy eating pizza.",
]

question = "How can I create a function in Python?"


document_embeddings = model.embed_documents(documents)
question_embedding = model.embed_query(question)


for document, embedding in zip(
    documents,
    document_embeddings,
):
    similarity = np.dot(
        question_embedding,
        embedding,
    )

    print("\nDocument:")
    print(document)

    print("Similarity:")
    print(round(float(similarity), 4))