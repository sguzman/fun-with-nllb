#!/usr/bin/env python3
# main.py — NLLB translation CLI with batching, tuning knobs, and logging
# Python 3.12+

import os
import sys
import argparse
import logging
import traceback
from typing import Tuple, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Be compatible with multiple huggingface_hub versions
try:
    from huggingface_hub.errors import HfHubHTTPError  # modern
except Exception:
    try:
        from huggingface_hub.utils._errors import HfHubHTTPError  # older
    except Exception:

        class HfHubHTTPError(Exception):
            pass


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_MODEL = "1.3B"  # choices: "3.3B", "1.3B", "600M" or a full repo id
DEFAULT_SRC = "spa_Latn"
DEFAULT_TGT = "eng_Latn"
DEFAULT_BEAMS = 4
DEFAULT_BATCH = 16
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_INPUT = "in.txt"
DEFAULT_OUTPUT = "out.txt"
DEFAULT_VERBOSITY = 1  # 0=warning, 1=info, 2=debug


# ----------------------------
# Helpers
# ----------------------------
def model_choice_to_repo(model_choice: str) -> str:
    """
    Map shorthand to the official repo id. If the user passes a repo id,
    return it unchanged.
    """
    mc = model_choice.strip()
    if "/" in mc:
        return mc  # already a repo id
    table = {
        "3.3B": "facebook/nllb-200-3.3B",
        "1.3B": "facebook/nllb-200-distilled-1.3B",
        "600M": "facebook/nllb-200-distilled-600M",
    }
    if mc not in table:
        raise ValueError(
            f"Unknown model choice '{mc}'. Use one of 3.3B, 1.3B, 600M, or pass a full repo id like 'facebook/nllb-200-3.3B'."
        )
    return table[mc]


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device() -> Tuple[str, torch.dtype | None]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", None


def anonymized_snapshot(repo_id: str) -> str:
    """
    Download/resolve a snapshot anonymously, even if a bad token is present.
    Returns local path suitable for from_pretrained(local, local_files_only=True).
    """
    saved = {}
    # Scrub possible token env vars for the duration of this call
    for k in ("HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN"):
        if k in os.environ:
            saved[k] = os.environ.pop(k)

    try:
        path = snapshot_download(
            repo_id=repo_id,
            token=None,
            resume_download=True,
            local_files_only=False,
        )
        return path
    finally:
        os.environ.update(saved)


def load_tok_and_model(
    repo_id: str, device: str, torch_dtype
) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str]:
    """
    Resolve snapshot anonymously and load tokenizer+model from local files only.
    Returns (tokenizer, model, resolved_local_path).
    """
    local = anonymized_snapshot(repo_id)
    tok = AutoTokenizer.from_pretrained(local, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local,
        local_files_only=True,
        torch_dtype=torch_dtype,
    ).to(device)
    return tok, model, local


def read_nonempty_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as g:
        for s in lines:
            g.write(s + "\n")


# ----------------------------
# Core translation
# ----------------------------
def translate_file(
    repo_id: str,
    src_lang: str,
    tgt_lang: str,
    beams: int,
    batch_size: int,
    max_new_tokens: int,
    infile: str,
    outfile: str,
    verbosity: int,
) -> None:
    configure_logging(verbosity)
    logger = logging.getLogger("nllb")

    device, torch_dtype = resolve_device()
    logger.info(
        f"Device: {device}  torch={torch.__version__}  CUDA={torch.cuda.is_available()}  dtype={torch_dtype}"
    )
    logger.info(f"Model repo: {repo_id}")
    logger.info(
        f"Params: beams={beams} batch={batch_size} max_new_tokens={max_new_tokens} src={src_lang} tgt={tgt_lang}"
    )
    logger.info(f"In: {infile}  Out: {outfile}")

    try:
        tok, model, local = load_tok_and_model(repo_id, device, torch_dtype)
    except HfHubHTTPError as e:
        logger.error("Hugging Face HTTP error while fetching the model.")
        logger.error(str(e))
        logger.error(
            "If you see 401 Unauthorized: unset bad tokens (unset HUGGINGFACE_HUB_TOKEN; huggingface-cli logout) "
            "or accept the model license on the repo page."
        )
        raise
    except Exception as e:
        logger.error("Failed to load tokenizer/model.")
        logger.error("".join(traceback.format_exception_only(type(e), e)).strip())
        raise

    logger.info(f"Resolved local snapshot: {local}")
    logger.info(f"Tokenizer path: {getattr(tok, 'name_or_path', '?')}")
    logger.info(
        f"Model path: {getattr(getattr(model, 'config', None), '_name_or_path', None) or getattr(model, 'name_or_path', '?')}"
    )

    # Configure languages
    tok.src_lang = src_lang
    tgt_id = tok.convert_tokens_to_ids(tgt_lang)
    if tgt_id is None or tgt_id == tok.unk_token_id:
        logging.warning(
            "Target language token id not found or is <unk>. Check your tgt_lang code (e.g., 'eng_Latn')."
        )

    # Read input
    lines = read_nonempty_lines(infile)
    logger.info(f"Loaded {len(lines)} non-empty line(s) from {infile}")

    # Translate in batches
    outputs: List[str] = []
    total = len(lines)
    for start in range(0, total, batch_size):
        batch = lines[start : start + batch_size]
        logger.debug(f"Batch {start // batch_size + 1}: size={len(batch)}")

        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,  # truncate to tokenizer max length
            # If you want to be explicit:
            # max_length=tok.model_max_length,
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                num_beams=beams,
                max_new_tokens=max_new_tokens,
                forced_bos_token_id=tgt_id,
            )

        decoded = tok.batch_decode(out, skip_special_tokens=True)
        outputs.extend(decoded)

        if verbosity >= 1:
            done = min(start + batch_size, total)
            logger.info(f"Translated {done}/{total}")

    write_lines(outfile, outputs)
    logger.info(f"Wrote {len(outputs)} line(s) to {outfile}")


# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nllb-translate",
        description="Translate lines from an input file using NLLB with configurable batching and decoding.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model shorthand (3.3B, 1.3B, 600M) or a full repo id (e.g., facebook/nllb-200-3.3B).",
    )
    p.add_argument(
        "--src", default=DEFAULT_SRC, help="Source language code (e.g., spa_Latn)."
    )
    p.add_argument(
        "--tgt", default=DEFAULT_TGT, help="Target language code (e.g., eng_Latn)."
    )
    p.add_argument(
        "--beams",
        type=int,
        default=DEFAULT_BEAMS,
        help="Beam search width (num_beams).",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Batch size (lines per forward pass).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens per line (not words).",
    )
    p.add_argument(
        "--infile",
        default=DEFAULT_INPUT,
        help="Input text file (one segment per non-empty line).",
    )
    p.add_argument(
        "--outfile",
        default=DEFAULT_OUTPUT,
        help="Output text file (one translation per line).",
    )
    p.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=DEFAULT_VERBOSITY,
        help="0=warning, 1=info, 2=debug.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        repo_id = model_choice_to_repo(args.model)
    except Exception as e:
        print(f"Bad --model value: {e}", file=sys.stderr)
        return 2

    try:
        translate_file(
            repo_id=repo_id,
            src_lang=args.src,
            tgt_lang=args.tgt,
            beams=args.beams,
            batch_size=args.batch,
            max_new_tokens=args.max_new_tokens,
            infile=args.infile,
            outfile=args.outfile,
            verbosity=args.verbosity,
        )
    except Exception:
        # Translate_file logs details; just return failure here.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
