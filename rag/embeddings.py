from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):

    texts = []

    for chunk in chunks:

        text = f"""
Category: {chunk['category']}

Title: {chunk['title']}

{chunk['text']}
"""

        texts.append(text)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


def create_query_embedding(query):

    embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    return embedding