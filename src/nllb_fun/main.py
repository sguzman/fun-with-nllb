import sys, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "facebook/nllb-200-3.3B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else None
).to("cuda" if torch.cuda.is_available() else "cpu")

src_lang, tgt_lang = "spa_Latn", "eng_Latn"
tok.src_lang = src_lang
tgt_id = tok.convert_tokens_to_ids(tgt_lang)

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
        g.writelines(s + "\n" for s in tok.batch_decode(out, skip_special_tokens=True))
