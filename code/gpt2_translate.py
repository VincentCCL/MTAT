#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 translation via prompting")

    parser.add_argument("--input", required=True, help="Input file (one sentence per line)")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="Dutch")

    parser.add_argument("--model", default="gpt2")

    parser.add_argument("--prompt-template", required=True,
                        help="Prompt template with {text}, {source_lang}, {target_lang}")

    parser.add_argument("--max-new-tokens", type=int, default=50)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--show-translations", action="store_true")
    parser.add_argument("--show-n", type=int, default=5)

    parser.add_argument("--strip-after-first-line", action="store_true")
    parser.add_argument("--strip-quotes", action="store_true")

    return parser.parse_args()


def build_prompt(template, text, source_lang, target_lang):
    return template.format(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang
    )


def clean_output(text, strip_after_first_line=False, strip_quotes=False):
    if strip_after_first_line:
        text = text.split("\n")[0]

    if strip_quotes:
        text = text.strip().strip('"').strip("'")

    return text.strip()


def generate_batch(model, tokenizer, prompts, args):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = args.temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            top_k=args.top_k if do_sample else None,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return decoded


def main():
    args = parse_args()

    print(f"[INFO] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()

    preds = []
    shown = 0

    print("[INFO] Translating...")

    for i in tqdm(range(0, len(lines), args.batch_size)):
        batch_lines = lines[i:i + args.batch_size]

        prompts = [
            build_prompt(args.prompt_template, line, args.source_lang, args.target_lang)
            for line in batch_lines
        ]

        outputs = generate_batch(model, tokenizer, prompts, args)

        for prompt, full_output in zip(prompts, outputs):
            continuation = full_output[len(prompt):].strip()

            cleaned = clean_output(
                continuation,
                strip_after_first_line=args.strip_after_first_line,
                strip_quotes=args.strip_quotes
            )

            preds.append(cleaned)

            if args.show_translations and shown < args.show_n:
                print("\n---")
                print("SRC:", prompt)
                print("HYP:", cleaned)
                shown += 1

    Path(args.output).write_text("\n".join(preds), encoding="utf-8")

    print(f"[INFO] Done. Saved to {args.output}")


if __name__ == "__main__":
    main()