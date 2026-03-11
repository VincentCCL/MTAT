#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


BASE_URL = "https://api.helmholtz-blablador.fz-juelich.de/v1/"


def get_api_key(api_key=None, api_env=None, kaggle_secret=None):
    if api_key:
        return api_key

    if api_env:
        key = os.environ.get(api_env)
        if key:
            return key
        raise ValueError(f"Environment variable {api_env} not set")

    if kaggle_secret:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        return client.get_secret(kaggle_secret)

    raise ValueError("No API key provided")


def build_prompt(sentences, src_lang, tgt_lang):
    payload = {"sentences": sentences}

    if src_lang:
        instruction = f"Translate the following sentences from {src_lang} to {tgt_lang}."
    else:
        instruction = f"Translate the following sentences to {tgt_lang}."

    instruction += (
        " Return ONLY valid JSON with key 'translations'. "
        "The value must be a list of translated sentences in the same order. "
        "Do not add explanations or markdown."
    )

    return instruction + "\n\n" + json.dumps(payload, ensure_ascii=False)


def parse_output(text, n):
    data = json.loads(text)
    translations = data["translations"]

    if len(translations) != n:
        raise ValueError(f"Translation length mismatch: expected {n}, got {len(translations)}")

    return [t.strip() for t in translations]


def translate_batch(
    client,
    sentences,
    model,
    src_lang,
    tgt_lang,
    temperature,
    max_retries,
    retry_wait,
    timeout,
):
    prompt = build_prompt(sentences, src_lang, tgt_lang)

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a machine translation system. Follow the requested output format exactly."},
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
            )

            text = response.choices[0].message.content
            if text is None:
                raise ValueError("Empty response content")

            return parse_output(text, len(sentences))

        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                break
            wait = retry_wait * (2 ** attempt)
            print(f"[warn] retrying batch after error: {e} (wait {wait:.1f}s)", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"Batch failed after {max_retries} attempts: {last_error}")


def chunk(data, size):
    for i in range(0, len(data), size):
        yield i, data[i:i + size]


def translate_file(args):
    api_key = get_api_key(args.api_key, args.api_env, args.kaggle_secret)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    with open(args.input, encoding="utf-8") as f:
        lines = [x.rstrip("\n") for x in f]

    output = list(lines)

    idx = []
    sentences = []

    for i, line in enumerate(lines):
        if line.strip():
            idx.append(i)
            sentences.append(line)

    batches = list(chunk(sentences, args.batch_size))

    if args.progress and tqdm is not None:
        batches_iter = tqdm(batches, desc="Translating", unit="batch")
    else:
        batches_iter = batches

    for batch_i, (start, batch) in enumerate(batches_iter, 1):
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
        )

        for j, t in enumerate(translations):
            output[idx[start + j]] = t

        if args.print_batches:
            print(f"\n--- Batch {batch_i} ---")
            for s, t in zip(batch, translations):
                print("SRC:", s)
                print("TGT:", t)
                print()

        elif args.progress_every and batch_i % args.progress_every == 0:
            print(f"[info] translated {start + len(batch)} sentences", file=sys.stderr)

        if args.save_every and batch_i % args.save_every == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                for line in output:
                    f.write(line + "\n")

    with open(args.output, "w", encoding="utf-8") as f:
        for line in output:
            f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--target-lang", required=True)
    ap.add_argument("--source-lang")

    ap.add_argument("--model", default="alias-fast")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-wait", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=120.0)

    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--print-batches", action="store_true")
    ap.add_argument("--save-every", type=int, default=0)

    key = ap.add_mutually_exclusive_group(required=True)
    key.add_argument("--api-key")
    key.add_argument("--api-env")
    key.add_argument("--kaggle-secret")

    args = ap.parse_args()
    translate_file(args)


if __name__ == "__main__":
    main()