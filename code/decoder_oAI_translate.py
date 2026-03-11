#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_BASE_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/"


def get_api_key(
    api_key: Optional[str] = None,
    api_env: Optional[str] = None,
    kaggle_secret: Optional[str] = None,
) -> str:
    if api_key:
        return api_key

    if api_env:
        value = os.environ.get(api_env)
        if value:
            return value
        raise ValueError(f"Environment variable '{api_env}' is not set")

    if kaggle_secret:
        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError as exc:
            raise ValueError(
                "Could not import kaggle_secrets. --kaggle-secret only works inside Kaggle."
            ) from exc

        client = UserSecretsClient()
        value = client.get_secret(kaggle_secret)
        if value:
            return value
        raise ValueError(f"Kaggle secret '{kaggle_secret}' not found or empty")

    raise ValueError("Provide one of: --api-key, --api-env, or --kaggle-secret")


def build_prompt(sentences: List[str], src_lang: Optional[str], tgt_lang: str) -> str:
    payload = {"sentences": sentences}

    if src_lang:
        instruction = f"Translate the following sentences from {src_lang} to {tgt_lang}. "
    else:
        instruction = f"Translate the following sentences to {tgt_lang}. "

    instruction += (
        "Return ONLY valid JSON with exactly one key: 'translations'. "
        "Its value must be a list of translated strings in exactly the same order "
        "and with exactly the same length as the input. "
        "Do not add explanations, comments, markdown, or extra text."
    )

    return instruction + "\n\nInput JSON:\n" + json.dumps(payload, ensure_ascii=False)


def parse_output(text: str, expected_n: int) -> List[str]:
    data = json.loads(text)

    if not isinstance(data, dict) or "translations" not in data:
        raise ValueError("Model output does not contain a 'translations' key")

    translations = data["translations"]

    if not isinstance(translations, list):
        raise ValueError("'translations' is not a list")

    if len(translations) != expected_n:
        raise ValueError(
            f"Translation length mismatch: expected {expected_n}, got {len(translations)}"
        )

    cleaned = []
    for item in translations:
        if not isinstance(item, str):
            raise ValueError("Each translation must be a string")
        cleaned.append(item.strip())

    return cleaned


def translate_batch(
    client: OpenAI,
    sentences: List[str],
    model: str,
    src_lang: Optional[str],
    tgt_lang: str,
    temperature: float,
    max_retries: int,
    retry_wait: float,
    timeout: float,
    debug: bool = False,
) -> List[str]:
    prompt = build_prompt(sentences, src_lang, tgt_lang)
    last_error = None

    for attempt in range(max_retries):
        try:
            if debug:
                print(
                    f"[debug] sending request for {len(sentences)} sentence(s)",
                    file=sys.stderr,
                    flush=True,
                )

            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a machine translation system. "
                            "Follow the requested output format exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
            )

            text = response.choices[0].message.content
            if text is None:
                raise ValueError("Empty response content")

            if debug:
                preview = text[:300].replace("\n", "\\n")
                print(f"[debug] raw response: {preview}", file=sys.stderr, flush=True)

            return parse_output(text, len(sentences))

        except Exception as e:
            last_error = e
            msg = str(e)

            if attempt == max_retries - 1:
                break

            if "502" in msg or "Proxy Error" in msg:
                wait = max(retry_wait * (2 ** attempt), 10.0)
            else:
                wait = retry_wait * (2 ** attempt)

            print(
                f"[warn] batch failed (attempt {attempt + 1}/{max_retries}): {e}",
                file=sys.stderr,
                flush=True,
            )
            print(f"[warn] retrying in {wait:.1f}s", file=sys.stderr, flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Batch failed after {max_retries} attempts: {last_error}")


def chunk_list(data: List[str], size: int):
    for i in range(0, len(data), size):
        yield i, data[i:i + size]


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def translate_file(args):
    api_key = get_api_key(args.api_key, args.api_env, args.kaggle_secret)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    all_lines = read_lines(args.input)
    output_lines = list(all_lines)

    nonempty_indices = []
    nonempty_sentences = []

    for i, line in enumerate(all_lines):
        if line.strip():
            nonempty_indices.append(i)
            nonempty_sentences.append(line)

    total_sentences = len(nonempty_sentences)
    if total_sentences == 0:
        write_lines(args.output, output_lines)
        print("[info] input contains no non-empty lines", file=sys.stderr, flush=True)
        return

    batches = list(chunk_list(nonempty_sentences, args.batch_size))

    if args.progress:
        if tqdm is None:
            raise ImportError("tqdm is not installed. Install it with: pip install tqdm")
        iterator = tqdm(batches, desc="Translating", unit="batch")
    else:
        iterator = batches

    done = 0

    for batch_no, (start, batch) in enumerate(iterator, start=1):
        if args.debug:
            print(
                f"[debug] starting batch {batch_no}/{len(batches)}",
                file=sys.stderr,
                flush=True,
            )

        translations = translate_batch(
            client=client,
            sentences=batch,
            model=args.model,
            src_lang=args.source_lang,
            tgt_lang=args.target_lang,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
            timeout=args.timeout,
            debug=args.debug,
        )

        for j, translation in enumerate(translations):
            original_idx = nonempty_indices[start + j]
            output_lines[original_idx] = translation

        done += len(batch)

        if args.print_batches:
            print(f"\n--- Batch {batch_no} ---", flush=True)
            for src, tgt in zip(batch, translations):
                print(f"SRC: {src}", flush=True)
                print(f"TGT: {tgt}", flush=True)
                print("", flush=True)

        if args.print_every > 0 and batch_no % args.print_every == 0:
            print(
                f"[info] translated {done}/{total_sentences} non-empty lines",
                file=sys.stderr,
                flush=True,
            )

        if args.save_every > 0 and batch_no % args.save_every == 0:
            write_lines(args.output, output_lines)
            print(
                f"[info] partial output saved after batch {batch_no}",
                file=sys.stderr,
                flush=True,
            )

    write_lines(args.output, output_lines)
    print(f"[info] finished, wrote output to {args.output}", file=sys.stderr, flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description="Translate a text file via an OpenAI-compatible endpoint.")

    ap.add_argument("--input", required=True, help="Input text file")
    ap.add_argument("--output", required=True, help="Output text file")
    ap.add_argument("--target-lang", required=True, help="Target language, e.g. Dutch")
    ap.add_argument("--source-lang", default=None, help="Source language, e.g. English")

    ap.add_argument("--model", required=True, help="Model id or alias")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    ap.add_argument("--batch-size", type=int, default=1, help="Number of sentences per request")
    ap.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    ap.add_argument("--max-retries", type=int, default=5, help="Retries per batch")
    ap.add_argument("--retry-wait", type=float, default=2.0, help="Initial retry wait in seconds")
    ap.add_argument("--timeout", type=float, default=30.0, help="Timeout per request in seconds")

    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bar")
    ap.add_argument("--print-batches", action="store_true", help="Print translations batch by batch")
    ap.add_argument("--print-every", type=int, default=0, help="Print progress every N batches")
    ap.add_argument("--save-every", type=int, default=0, help="Save partial output every N batches")
    ap.add_argument("--debug", action="store_true", help="Print debug messages")

    key_group = ap.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--api-key", help="Pass API key directly")
    key_group.add_argument("--api-env", help="Read API key from environment variable")
    key_group.add_argument("--kaggle-secret", help="Read API key from Kaggle secret")

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    translate_file(args)


if __name__ == "__main__":
    main()