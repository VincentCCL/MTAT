#!/usr/bin/env python3
# qwen_translate.py

import argparse
import math
import os
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate a text file with a decoder-only Qwen model."
    )

    parser.add_argument("--input", required=True, help="Input text file (one sentence per line).")
    parser.add_argument("--output", required=True, help="Output text file for translations.")
    parser.add_argument("--target-lang", required=True, help="Target language name, e.g. Dutch.")
    parser.add_argument("--source-lang", default="English", help="Source language name, e.g. English.")
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name, e.g. Qwen/Qwen2.5-1.5B-Instruct"
    )

    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam size for generation.")
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=1024,
        help="Maximum tokenized prompt length."
    )

    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("--print-batches", action="store_true", help="Print batch progress.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print progress every N lines translated (0 = never)."
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save intermediate output every N lines (0 = only save at end)."
    )
    parser.add_argument("--debug", action="store_true", help="Verbose debugging output.")
    parser.add_argument(
        "--show-translations",
        action="store_true",
        help="Print some translations while running."
    )
    parser.add_argument(
        "--show-n",
        type=int,
        default=3,
        help="How many translations to show when --show-translations is enabled."
    )

    parser.add_argument(
        "--prompt-template",
        default="Translate the following text from {source_lang} to {target_lang}. "
                "Only output the translation.\n\n{text}",
        help=(
            "Custom prompt template. Available placeholders: "
            "{source_lang}, {target_lang}, {text}"
        )
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a careful machine translation system.",
        help="Optional system prompt used when --use-chat-template is enabled."
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap prompts using the tokenizer chat template when available."
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model."
    )

    return parser.parse_args()


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def chunk_list(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def build_prompt(text: str, source_lang: str, target_lang: str, template: str) -> str:
    return template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        text=text
    )


def maybe_apply_chat_template(tokenizer, user_prompt: str, system_prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return user_prompt

    if not hasattr(tokenizer, "apply_chat_template"):
        return user_prompt

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def clean_output(text: str) -> str:
    return text.strip().replace("\n", " ").strip()


def main():
    args = parse_args()

    if args.temperature > 0 and args.num_beams > 1:
        raise ValueError("Use either sampling (temperature > 0) or beam search (num_beams > 1), not both.")

    if args.debug:
        print("[DEBUG] Loading input...")
    src_lines = read_lines(args.input)

    if args.debug:
        print(f"[DEBUG] Loaded {len(src_lines)} lines from {args.input}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.debug:
        print(f"[DEBUG] Using device: {device}")

    if args.debug:
        print(f"[DEBUG] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.debug:
        print(f"[DEBUG] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=args.trust_remote_code
    )
    model.to(device)
    model.eval()

    batches = chunk_list(src_lines, args.batch_size)
    outputs = []

    iterator = batches
    if args.progress:
        iterator = tqdm(batches, desc="Translating", unit="batch")

    total_done = 0

    for batch_idx, batch in enumerate(iterator, start=1):
        prompts = []
        for line in batch:
            prompt = build_prompt(
                text=line,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                template=args.prompt_template
            )
            prompt = maybe_apply_chat_template(
                tokenizer=tokenizer,
                user_prompt=prompt,
                system_prompt=args.system_prompt,
                use_chat_template=args.use_chat_template
            )
            prompts.append(prompt)

        if args.debug and batch_idx == 1:
            print("\n[DEBUG] First prompt example:\n")
            print(prompts[0])
            print()

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if args.num_beams > 1:
            gen_kwargs["num_beams"] = args.num_beams
            gen_kwargs["do_sample"] = False
        elif args.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
        else:
                gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated = model.generate(**enc, **gen_kwargs)

        batch_out = []
        for i in range(len(batch)):
            gen_tokens = generated[i][prompt_lengths[i]:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            text = clean_output(text)
            batch_out.append(text)

        outputs.extend(batch_out)
        total_done += len(batch)

        if args.print_batches:
            print(f"[INFO] Finished batch {batch_idx}/{len(batches)}")

        if args.print_every and total_done % args.print_every == 0:
            print(f"[INFO] Translated {total_done}/{len(src_lines)} lines")

        if args.show_translations:
            n_show = min(args.show_n, len(batch))
            print(f"\n=== Sample translations from batch {batch_idx} ===")
            for j in range(n_show):
                print(f"SRC: {batch[j]}")
                print(f"HYP: {batch_out[j]}")
                print()

        if args.save_every and total_done % args.save_every == 0:
            if args.debug:
                print(f"[DEBUG] Saving intermediate output after {total_done} lines...")
            write_lines(args.output, outputs)

    write_lines(args.output, outputs)
    print(f"[DONE] Wrote {len(outputs)} translations to {args.output}")


if __name__ == "__main__":
    main()