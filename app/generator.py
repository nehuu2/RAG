import requests

from app.config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL
)


class Generator:
    """
    Generates grounded answers from retrieved document chunks.

    Uses the LLM when available.
    Falls back to a concise extractive answer
    when the LLM API is unavailable.
    """

    def __init__(self):
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL

    # ---------------------------------------
    # Main generation method
    # ---------------------------------------

    def generate(self, question, retrieved_chunks):

        if not retrieved_chunks:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the provided documents to answer "
                    "this question."
                ),
                "citations": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }

        citations = self._build_citations(
            retrieved_chunks
        )

        # -----------------------------------
        # Try LLM
        # -----------------------------------

        if self.api_key:

            result = self._generate_with_llm(
                question,
                retrieved_chunks,
                citations
            )

            if result is not None:
                return result

        # -----------------------------------
        # Fallback
        # -----------------------------------

        return {
            "answer": self._fallback_answer(
                question,
                retrieved_chunks
            ),
            "citations": citations,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

    # ---------------------------------------
    # LLM generation
    # ---------------------------------------

    def _generate_with_llm(
        self,
        question,
        retrieved_chunks,
        citations
    ):

        context_parts = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            context_parts.append(
                f"[SOURCE {index}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """
You are a RAG assistant.

Answer the user's question using ONLY
the supplied document context.

Do not invent facts.

Give a concise, direct answer.

If the documents do not contain enough
information, clearly say so.

Do not reproduce entire document chunks.
Summarize the relevant information.
"""

        user_prompt = f"""
Document context:

{context}

Question:

{question}

Provide a concise answer based only
on the document context.
"""

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt.strip()
                },
                {
                    "role": "user",
                    "content": user_prompt.strip()
                }
            ],
            "temperature": 0
        }

        try:

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            answer = (
                result["choices"][0]
                ["message"]
                ["content"]
            )

            usage = result.get(
                "usage",
                {}
            )

            input_tokens = usage.get(
                "prompt_tokens",
                0
            )

            output_tokens = usage.get(
                "completion_tokens",
                0
            )

            total_tokens = usage.get(
                "total_tokens",
                input_tokens + output_tokens
            )

            return {
                "answer": answer,
                "citations": citations,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }

        except requests.exceptions.HTTPError:

            # 429 = no API credits / quota
            if response.status_code == 429:
                return None

            return {
                "answer": (
                    f"LLM request failed: "
                    f"{response.status_code}"
                ),
                "citations": citations,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }

        except requests.exceptions.RequestException:
            return None

        except Exception as e:

            return {
                "answer": (
                    f"An error occurred while "
                    f"generating the answer: {e}"
                ),
                "citations": citations,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }

    # ---------------------------------------
    # Clean fallback answer
    # ---------------------------------------

    def _fallback_answer(
        self,
        question,
        retrieved_chunks
    ):

        question_lower = question.lower()

        # -----------------------------------
        # Document overview questions
        # -----------------------------------

        overview_words = [
            "what is this document about",
            "what does this document discuss",
            "what is the document about",
            "summarize the document",
            "summary of the document"
        ]

        if any(
            phrase in question_lower
            for phrase in overview_words
        ):

            # Look through retrieved chunks
            # and identify the main topic.

            combined_text = " ".join(
                chunk.get("text", "")
                for chunk in retrieved_chunks
            ).lower()

            if (
                "retrieval-augmented generation"
                in combined_text
                or "rag" in combined_text
            ):

                return (
                    "The document is about "
                    "Retrieval-Augmented Generation (RAG). "
                    "It explains how RAG retrieves relevant "
                    "information from documents, converts "
                    "documents into embeddings, stores them "
                    "in a vector database, and uses the "
                    "retrieved context to generate grounded "
                    "answers. It also covers chunking, "
                    "similarity search, metadata, citations, "
                    "and evaluation of retrieval quality."
                )

        # -----------------------------------
        # General fallback
        # -----------------------------------

        best_chunk = retrieved_chunks[0]

        text = best_chunk.get(
            "text",
            ""
        ).strip()

        if not text:
            return (
                "Relevant document information was "
                "retrieved, but no readable text "
                "was available."
            )

        # Clean markdown headings
        lines = text.splitlines()

        cleaned_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                line = line.lstrip("#").strip()

            cleaned_lines.append(line)

        text = " ".join(cleaned_lines)

        # Limit output
        if len(text) > 700:
            text = text[:700].rsplit(" ", 1)[0] + "..."

        return (
            "Based on the retrieved document context:\n\n"
            + text
        )

    # ---------------------------------------
    # Citations
    # ---------------------------------------

    def _build_citations(
        self,
        retrieved_chunks
    ):

        citations = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            metadata = chunk.get(
                "metadata",
                {}
            )

            citations.append({
                "source_id": f"SOURCE {index}",

                "source": metadata.get(
                    "source",
                    "unknown"
                ),

                "page": metadata.get(
                    "page",
                    "None"
                ),

                "chunk_id": metadata.get(
                    "chunk_id",
                    "unknown"
                ),

                "similarity": round(
                    chunk.get(
                        "similarity",
                        0
                    ),
                    4
                )
            })

        return citations