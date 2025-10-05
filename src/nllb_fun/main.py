import os
import sys
import traceback
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download, HfHubHTTPError

# Choose a default model. You can override with env MODEL_ID
MODEL_ID = os.environ.get("MODEL_ID", "facebook/nllb-200-3.3B")
SRC_LANG = os.environ.get("SRC_LANG", "spa_Latn")
TGT_LANG = os.environ.get("TGT_LANG", "eng_Latn")


def load_snapshot_locally(model_id: str) -> str:
    """
    Download (or find in cache) the model snapshot anonymously.
    Returns a local path suitable for from_pretrained().
    """
    # Force anonymous: ensure no bad token gets sent
    env = "HUGGINGFACE_HUB_TOKEN"
    saved = os.environ.pop(env, None)
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_files_only=False,
            resume_download=True,
            # force download of refs without using any token
            token=None,
        )
        return path
    finally:
        if saved is not None:
            os.environ[env] = saved


def load_tok_and_model(model_id: str, device: str):
    """
    Try anonymous first; if 401, suggest distilled model or valid login.
    Also supports MODEL_DIR if you already have a local snapshot.
    """
    model_dir = os.environ.get("MODEL_DIR")
    try_ids = []

    if model_dir:
        try_ids.append(model_dir)  # explicit local path
    try_ids.append(model_id)  # requested repo id

    # Soft fallback for convenience if user typed the big one
    if model_id == "facebook/nllb-200-3.3B":
        try_ids.append("facebook/nllb-200-distilled-1.3B")
        try_ids.append("facebook/nllb-200-distilled-600M")

    last_err = None
    for mid in try_ids:
        try:
            # If a local path, do not hit the hub at all
            if os.path.isdir(mid):
                tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    mid, local_files_only=True
                )
                return tok, model.to(device)
            # Otherwise, make sure the snapshot is present anonymously
            local = load_snapshot_locally(mid)
            tok = AutoTokenizer.from_pretrained(local, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                local,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            ).to(device)
            return tok, model
        except HfHubHTTPError as e:
            last_err = e
            # 401 -> invalid creds were sent somewhere or the repo is gated
            if "401" in str(e):
                print(
                    "Hub returned 401. Either a bad token is set or the repo is gated.",
                    file=sys.stderr,
                )
        except Exception as e:
            last_err = e

    # If we reach here, give a clear message
    raise RuntimeError(
        f"Could not load model. Last error: {last_err}\n"
        "Tips:\n"
        "- Ensure no invalid HUGGINGFACE_HUB_TOKEN is set (unset it or `huggingface-cli logout`).\n"
        "- Try distilled models: facebook/nllb-200-distilled-1.3B or ...-600M.\n"
        "- To work offline: prefetch with `huggingface-cli download <repo> --local-dir ./models/nllb` "
        "and set MODEL_DIR=./models/nllb.\n"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Device: {device}  torch={torch.__version__}  CUDA={torch.cuda.is_available()}"
    )

    try:
        tok, model = load_tok_and_model(MODEL_ID, device)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    tok.src_lang = SRC_LANG
    tgt_id = tok.convert_tokens_to_ids(TGT_LANG)

    with open("in.txt", "r", encoding="utf-8") as f, open(
        "out.txt", "w", encoding="utf-8"
    ) as g:
        lines = [l.strip() for l in f if l.strip()]
        for i in range(0, len(lines), 16):
            batch = lines[i : i + 16]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to(
                model.device
            )
            out = model.generate(
                **inputs, forced_bos_token_id=tgt_id, max_new_tokens=256, num_beams=4
            )
            g.writelines(
                s + "\n" for s in tok.batch_decode(out, skip_special_tokens=True)
            )
    print("Wrote out.txt")


if __name__ == "__main__":
    main()
