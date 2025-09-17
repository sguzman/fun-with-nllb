{
  description = "NLLB translation playground (PyTorch+Transformers and CTranslate2) with CUDA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        # CUDA + unfree are required for NVIDIA libs and pytorch-bin
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      python = pkgs.python312;

      # CUDA-enabled CTranslate2 C++ runtime
      ctranslate2-cuda = pkgs.ctranslate2.override {withCUDA = true;};

      pythonEnv = python.withPackages (
        ps: let
          # Prefer upstream wheel-based PyTorch with CUDA if present; otherwise fall back.
          pytorchWithCuda =
            if builtins.hasAttr "pytorch-bin" ps
            then ps.pytorch-bin
            else (ps.pytorch.override {cudaSupport = true;});
        in [
          pytorchWithCuda
          ps.transformers
          ps.sentencepiece
          ps.accelerate
          ps.huggingface-hub
          ps.safetensors
          ps.pip
          ps.setuptools
          ps.wheel

          # Python bindings + converter CLI; link to CUDA runtime above
          (ps.ctranslate2.override {ctranslate2 = ctranslate2-cuda;})
        ]
      );
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          pythonEnv
          ctranslate2-cuda
          pkgs.git
        ];

        # Useful in many setups; harmless if already set
        NIXPKGS_ALLOW_UNFREE = "1";

        shellHook = ''
                      echo "✓ Python: $(python --version)"
                      python - <<'PY'
          import torch, shutil
          print("✓ torch", torch.__version__, "CUDA available:", torch.cuda.is_available())
          if torch.cuda.is_available():
              print("  GPU:", torch.cuda.get_device_name(0))
          try:
              import ctranslate2
              print("✓ ctranslate2", ctranslate2.__version__,
                    "converter:", shutil.which("ct2-transformers-converter") is not None)
          except Exception as e:
              print("ctranslate2 import failed:", e)
          PY
                      cat <<'TXT'

          Examples you can paste:

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
          tok = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="spa_Latn")
          tr  = ctranslate2.Translator("nllb-ct2", device="cuda")
          src = tok.convert_ids_to_tokens(tok.encode("hola mundo"))
          res = tr.translate_batch([src], target_prefix=[["eng_Latn"]])
          toks = res[0].hypotheses[0][1:]  # drop leading lang token
          print(tok.decode(tok.convert_tokens_to_ids(toks)))
          PY
          TXT
        '';
      };
    });
}
