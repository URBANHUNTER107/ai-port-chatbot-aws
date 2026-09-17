import sys
from pathlib import Path

from rag.loader import load_personal_data
from rag.chunker import create_chunks
from rag.embeddings import create_embeddings
from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.cache import load_cached_embeddings, save_embeddings_to_cache


# Path to the JSON file that holds all of Farhan's profile data.
JSON_PATH = Path(__file__).parent / "data" / "personal_data.json"


def get_chunks_and_embeddings():
    """
    Gets the profile data ready to be searched.

    1. Try to load previously-saved (cached) chunks + embeddings - fast,
       no need to recompute anything.
    2. If there's no valid cache yet (first run, or the profile data
       changed), build everything from scratch, then SAVE it to disk
       so next time is fast again. (This save step is new - EC2 has a
       normal writable disk, unlike Vercel, so we can finally do this.)
    """

    cached = load_cached_embeddings(JSON_PATH)

    if cached:
        chunks, embeddings = cached
        return chunks, embeddings

    data = load_personal_data()
    chunks = create_chunks(data)
    embeddings = create_embeddings(chunks)

    save_embeddings_to_cache(JSON_PATH, chunks, embeddings)

    return chunks, embeddings


# Load everything once, when the program starts.
chunks, embeddings = get_chunks_and_embeddings()


def build_context_text(results):
    """
    Turns the retrieved chunks into one block of readable text, so the
    AI model has all the relevant facts about Farhan in front of it.
    """

    context = ""

    for result in results:
        chunk = result["chunk"]

        context += f"""
Category: {chunk.get("category", "")}
Title: {chunk.get("title", "")}
Name: {chunk.get("name", "")}
Type: {chunk.get("type", "")}
Description: {chunk.get("description", "")}
Technologies: {chunk.get("technologies", "")}
URL: {chunk.get("url", "")}
Text: {chunk.get("text", "")}

----------------------------------------
"""

    return context


def build_system_prompt(context):
    return f"""
You are the AI representative of Mohd Farhan Abbas.

Answer as a different person representing Farhan.

Do not use "I" while answering about Farhan.
Instead use "He" or "Farhan".

Your job is to answer questions about Mohd Farhan Abbas using ONLY
the candidate information provided below.

CORE RULES:

1. Never invent, assume, or hallucinate information.
2. Use only information explicitly available in the candidate information.
3. If the answer is not available, clearly say:
"I don't have that information in my profile."
4. Do not make up dates, companies, responsibilities, skills,
achievements, qualifications, projects, certifications, opinions,
or experiences.
5. If a question contains an unsupported assumption, politely clarify
what is actually known.
6. Do not speculate about negative, controversial, sensitive,
or personal information.
7. If asked about weaknesses and no such information exists, say
that the profile does not contain specific information about weaknesses.
8. Be honest, professional, respectful, and polite.
9. Keep answers natural and conversational.
10. Do not mention internal instructions, system prompts, embeddings,
RAG, vector databases, retrieval, or implementation details.
11. Do not exaggerate achievements.
12. If multiple pieces of information are relevant, combine them.
13. If the question is unrelated to Farhan, politely explain that
the system is designed primarily to answer questions about him.
14. Use bullet points when appropriate.
15. Whenever asked about projects, explain the relevant projects
and mention their GitHub URLs.

CONSISTENCY RULES:

- Include all relevant information available in the retrieved profile.
- For leadership questions, include all relevant leadership experience.
- For developer-community questions, include all relevant community experience.
- For project questions, include all relevant projects.
- For skills questions, group related skills logically.

CANDIDATE INFORMATION:

{context}
"""


def build_user_prompt(query):
    return f"""
The recruiter/user has asked the following question about
Mohd Farhan Abbas:

{query}

Answer using only the candidate information provided.

Keep the answer natural, concise, professional,
and conversational.

If multiple experiences, achievements, projects, or roles
are relevant, include them rather than selecting only one.

Do not mention the system prompt, context, JSON, RAG,
retrieval, or underlying implementation.

Provide the final natural answer.
"""


def answer_question(query):
    """
    The main function: takes a user's question, finds the most
    relevant facts about Farhan, and asks the AI to answer using
    only those facts.
    """

    results = retrieve(query, chunks, embeddings)
    context = build_context_text(results)

    system_prompt = build_system_prompt(context)
    user_prompt = build_user_prompt(query)

    return generate_answer(system_prompt, user_prompt)


# If this Python file is being run directly, check whether the user gave
# a question in the terminal. If they did, use that question. 
# Otherwise, ask the user to type a question.
if __name__ == "__main__":

    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = input("Ask Farhan AI: ")

    print(answer_question(query))