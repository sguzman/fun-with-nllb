{
  description = "NLLB + Transformers + CTranslate2 on CUDA (nixpkgs 25.05)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      # Enable CUDA for the C++ ctranslate2 build (used by the Python bindings)
      overlays = [
        (final: prev: {
          ctranslate2 = prev.ctranslate2.override {withCUDA = true;};
        })
      ];

      pkgs = import nixpkgs {
        inherit system overlays;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      py = pkgs.python312;

      pythonEnv = py.withPackages (ps: [
        (ps.pytorch.override {cudaSupport = true;})
        ps.transformers
        ps.sentencepiece
        ps.accelerate
        ps.huggingface-hub
        ps.safetensors
        ps.ctranslate2 # Python package; links to CUDA-enabled pkgs.ctranslate2
      ]);
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          pythonEnv
          pkgs.ctranslate2 # CUDA build from overlay
          pkgs.git
        ];

        NIXPKGS_ALLOW_UNFREE = "1";

        shellHook = ''
                      echo "--- CUDA/NLP sanity check (25.05) ---"
                      python - <<'PY'
          import torch, shutil
          print("torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
          if torch.cuda.is_available():
              print("GPU:", torch.cuda.get_device_name(0))
          try:
              import ctranslate2 as c2
              print("ctranslate2:", getattr(c2, "__version__", "unknown"),
                    "CUDA devices:", getattr(c2, "get_cuda_device_count", lambda: "n/a")())
          except Exception as e:
              print("ctranslate2 import failed:", e)
          print("converter present:", shutil.which("ct2-transformers-converter") is not None)
          PY
                      cat <<'TXT'

          Examples:

          # 1) Transformers (spa -> eng)
          python - <<'PY'
          from transformers import pipeline
          pipe = pipeline("translation",
                          model="facebook/nllb-200-distilled-1.3B",
                          src_lang="spa_Latn", tgt_lang="eng_Latn")
          print(pipe("¿Cómo estás?")[0]["translation_text"])
          PY

          # 2) Convert NLLB to CTranslate2 + run on CUDA
          ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-ct2
          python - <<'PY'
          import ctranslate2, transformers
          SRC, TGT = "spa_Latn", "eng_Latn"
          tok = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=SRC)
          tr  = ctranslate2.Translator("nllb-ct2", device="cuda", compute_type="float16")
          src = tok.convert_ids_to_tokens(tok.encode("hola mundo"))
          res = tr.translate_batch([src], target_prefix=[[TGT]])
          toks = res[0].hypotheses[0][1:]  # drop leading lang token
          print(tok.decode(tok.convert_tokens_to_ids(toks)))
          PY
          TXT
        '';
      };
    });
}
