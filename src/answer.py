
"""LLM answer step using NVIDIA NIM and Llama 3.2 90B Vision."""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env", override=True)

K = 8
REFUSAL = "I can't find this in the documents."

MODEL = os.getenv(
    "NVIDIA_MODEL",
    "meta/llama-3.2-90b-vision-instruct"
).strip()
PROVIDER = "NVIDIA"

BASE_URL = "https://integrate.api.nvidia.com/v1"

INSTRUCTIONS = f"""You answer questions using ONLY the numbered sources below.
- Prefer current documents. If a source is marked OUTDATED, say so and do not treat it as current.
- JoSAA rules override other documents for counselling and seat allotment.
- Cite sources like [1], [2] after each claim.
- If the sources do not contain the answer, reply with exactly: {REFUSAL}
- Keep the answer under 80 words."""


class StopRun(Exception):
    """Raised when retrying cannot fix the error."""


DailyQuotaExceeded = StopRun  # Compatibility with existing code


def build_prompt(question, chunks):
    parts = [INSTRUCTIONS, "SOURCES:"]

    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] (file: {c['doc']}, PDF page {c['page']})\n"
            f"{c['text']}"
        )

    parts.append(f"QUESTION: {question}\nANSWER:")
    return "\n\n".join(parts)


def make_client():
    key = os.getenv("NVIDIA_API_KEY", "").strip().strip('"').strip("'")

    if not key:
        sys.exit(f"NVIDIA_API_KEY is empty or missing in {BASE / '.env'}")

    if not key.startswith("nvapi-"):
        sys.exit(
            "NVIDIA_API_KEY should be your NVIDIA API Catalog secret key, "
            "usually starting with 'nvapi-'. Check the key in "
            "https://build.nvidia.com/settings/api-keys"
        )

    if not MODEL:
        sys.exit("NVIDIA_MODEL is empty in .env")

    return OpenAI(
        api_key=key,
        base_url=BASE_URL,
        timeout=90.0,
        max_retries=0
    )


def _call(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=300
    )

    return (
        response.choices[0].message.content or ""
    ).strip()


def ask(client, prompt, retries=4):
    """Return the generated answer or stop on non-retryable errors."""

    for attempt in range(retries):
        try:
            answer = _call(client, prompt)

            if answer:
                return answer

            err = "empty answer"

        except Exception as e:
            err = str(e)
            low = err.lower()

            # Authentication, access, billing and quota errors
            # generally will not be fixed by retrying.
            permanent_errors = (
                "401",
                "403",
                "402",
                "unauthorized",
                "invalid api key",
                "invalid jwt",
                "insufficient",
                "no remaining credits",
                "perday",
                "quota exceeded",
                "payment required",
            )

            if any(term in low for term in permanent_errors):
                raise StopRun(err) from e

        if attempt < retries - 1:
            wait = 10 * (attempt + 1)
            print(
                f"  Temporary problem: {err[:200]} "
                f"-- retrying in {wait}s"
            )
            time.sleep(wait)

    return f"ERROR: {err}"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        sys.exit('Usage: python src/answer.py "your question"')

    from retriever import Retriever

    chunks = Retriever().retrieve(question, k=K)

    if not chunks:
        print(REFUSAL)
        sys.exit(0)

    try:
        client = make_client()
        print(ask(client, build_prompt(question, chunks)))

    except StopRun as e:
        sys.exit(f"Stopped: {e}")

    print("\nSources used:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c['doc']}, page {c['page']}")



