#!/usr/bin/env python3
"""
Translate a text file with the Blablador endpoint.

Features
--------
- Reads a plain text file, typically one sentence per line
- Preserves empty lines
- Supports batching
- Supports retries with exponential backoff
- Writes one translated line per input line
- Can optionally use a Kaggle secret or environment variable for the API key

Example
-------
python blablador_translate.py \
  --input test.en \
  --output test.nl \
  --target-lang Dutch \
  --source-lang English \
  --model alias-fast \
  --batch-size 8

Kaggle usage
------------
If you are in Kaggle, you can do either:
1. pass --api-key directly
2. store a Kaggle secret and use --kaggle-secret blablador
3. expose an env var first and use --api-env BLABLADOR_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI


def get_api_key(
    api_key: Optional[str] = None,
    api_env: Optional[str] = None,
    kaggle_secret: Optional[str] = None,
) -> str:
    """
    Resolve the API key from one of:
    - explicit --api-key
    - environment variable named by --api-env
    - Kaggle secret named by --kaggle-secret
    """
    if api_key:
        return api_key

    if api_env:
        value = os.environ.get(api_env)
        if value:
            return value
        raise ValueError(f"Environment variable '{api_env}' is not set.")

    if kaggle_secret:
        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError as exc:
            raise ValueError(
                "Could not import kaggle_secrets. Are you running this in Kaggle?"
            ) from exc

        client = UserSecretsClient()
        value = client.get_secret(kaggle_secret)
        if value:
            return value
        raise ValueError(f"Kaggle secret '{kaggle_secret}' was not found or is empty.")

    raise ValueError(
        "No API key provided. Use one of: --api-key, --api-env, or --kaggle-secret."
    )


def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def build_prompt(
    texts: List[str],
    target_language: str,
    source_language: Optional[str] = None,
) -> str:
    """
    Build a prompt that asks the model to translate a batch of sentences and
    return strict JSON so parsing is more reliable.
    """
    payload = {"sentences": texts}

    if source_language:
        instruction = (
            f"Translate each sentence from {source_language} to {target_language}. "
            "Return only valid JSON with exactly one key: 'translations'. "
            "The value must be a list of translated strings in the same order "
            "and of the same length as the input sentences. "
            "Do not add explanations, notes, or markdown."
        )
    else:
        instruction = (
            f"Translate each sentence to {target_language}. "
            "Return only valid JSON with exactly one key: 'translations'. "
            "The value must be a list of translated strings in the same order "
            "and of the same length as the input sentences. "
            "Do not add explanations, notes, or markdown."
        )

    return instruction + "\n\nInput JSON:\n" + json.dumps(payload, ensure_ascii=False)


def parse_translations(raw_text: str, expected_n: int) -> List[str]:
    """
    Parse model output as JSON and validate shape.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model output was not valid JSON. Output starts with:\n{raw_text[:500]}"
        ) from exc

    if not isinstance(data, dict) or "translations" not in data:
        raise ValueError(
            f"Model output did not contain the expected 'translations' key. "
            f"Output starts with:\n{raw_text[:500]}"
        )

    translations = data["translations"]
    if not isinstance(translations, list):
        raise ValueError("'translations' is not a list.")

    if len(translations) != expected_n:
        raise ValueError(
            f"Expected {expected_n} translations, got {len(translations)}."
        )

    cleaned: List[str] = []
    for item in translations:
        if not isinstance(item, str):
            raise ValueError("Each translation must be a string.")
        cleaned.append(item.strip())

    return cleaned


def translate_batch(
    client: OpenAI,
    texts: List[str],
    model: str,
    target_language: str,
    source_language: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_wait: float = 2.0,
) -> List[str]:
    """
    Translate a batch with retries.
    """
    prompt = build_prompt(
        texts=texts,
        target_language=target_language,
        source_language=source_language,
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a machine translation system. "
                            "Follow the format instructions exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response content from model.")

            return parse_translations(content, expected_n=len(texts))

        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_time = retry_wait * (2 ** (attempt - 1))
            print(
                f"[warn] batch failed on attempt {attempt}/{max_retries}: {exc}",
                file=sys.stderr,
            )
            print(f"[warn] retrying in {sleep_time:.1f}s", file=sys.stderr)
            time.sleep(sleep_time)

    raise RuntimeError(f"Batch failed after {max_retries} attempts: {last_error}")


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def chunk_indices(items: List[str], batch_size: int) -> List[tuple[int, int]]:
    return [(i, min(i + batch_size, len(items))) for i in range(0, len(items), batch_size)]


def translate_file(
    input_file: str,
    output_file: str,
    api_key: str,
    target_language: str,
    source_language: Optional[str] = None,
    model: str = "alias-fast",
    batch_size: int = 8,
    temperature: float = 0.0,
    base_url: str = "https://api.helmholtz-blablador.fz-juelich.de/v1/",
    max_retries: int = 5,
    retry_wait: float = 2.0,
    progress_every: int = 1,
) -> None:
    """
    Translate a file while preserving empty lines.
    Non-empty lines are translated in batches.
    """
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    client = make_client(api_key=api_key, base_url=base_url)

    all_lines = read_lines(input_file)
    output_lines = list(all_lines)

    nonempty_positions: List[int] = []
    nonempty_texts: List[str] = []

    for idx, line in enumerate(all_lines):
        if line.strip():
            nonempty_positions.append(idx)
            nonempty_texts.append(line)

    total = len(nonempty_texts)
    if total == 0:
        write_lines(output_file, output_lines)
        return

    batches = chunk_indices(nonempty_texts, batch_size)

    for batch_no, (start, end) in enumerate(batches, start=1):
        batch_texts = nonempty_texts[start:end]
        translations = translate_batch(
            client=client,
            texts=batch_texts,
            model=model,
            target_language=target_language,
            source_language=source_language,
            temperature=temperature,
            max_retries=max_retries,
            retry_wait=retry_wait,
        )

        for rel_idx, translated in enumerate(translations):
            original_pos = nonempty_positions[start + rel_idx]
            output_lines[original_pos] = translated

        if progress_every > 0 and (batch_no % progress_every == 0 or batch_no == len(batches)):
            done = end
            print(
                f"[info] translated {done}/{total} non-empty lines "
                f"({batch_no}/{len(batches)} batches)",
                file=sys.stderr,
            )

    write_lines(output_file, output_lines)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Translate a text file using the Blablador endpoint."
    )
    ap.add_argument("--input", required=True, help="Input text file")
    ap.add_argument("--output", required=True, help="Output text file")
    ap.add_argument("--target-lang", required=True, help="Target language, e.g. Dutch")
    ap.add_argument("--source-lang", default=None, help="Source language, e.g. English")
    ap.add_argument("--model", default="alias-fast", help="Model id or alias")
    ap.add_argument("--batch-size", type=int, default=8, help="Sentences per API call")
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="Generation temperature"
    )
    ap.add_argument(
        "--base-url",
        default="https://api.helmholtz-blablador.fz-juelich.de/v1/",
        help="Blablador API base URL",
    )
    ap.add_argument(
        "--max-retries", type=int, default=5, help="Maximum retries per batch"
    )
    ap.add_argument(
        "--retry-wait",
        type=float,
        default=2.0,
        help="Initial wait time for retry backoff in seconds",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N batches",
    )

    key_group = ap.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--api-key", help="API key passed directly")
    key_group.add_argument(
        "--api-env",
        help="Read API key from this environment variable, e.g. BLABLADOR_API_KEY",
    )
    key_group.add_argument(
        "--kaggle-secret",
        help="Read API key from this Kaggle secret name, e.g. blablador",
    )

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    api_key = get_api_key(
        api_key=args.api_key,
        api_env=args.api_env,
        kaggle_secret=args.kaggle_secret,
    )

    translate_file(
        input_file=args.input,
        output_file=args.output,
        api_key=api_key,
        target_language=args.target_lang,
        source_language=args.source_lang,
        model=args.model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        base_url=args.base_url,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()