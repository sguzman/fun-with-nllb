{
  description = "NLLB + Transformers + CTranslate2 on CUDA (dev shell)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      # Enable CUDA on the C++ ctranslate2 package globally
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

      pythonEnv = py.withPackages (ps: let
        torch =
          if ps ? pytorch-bin
          then ps.pytorch-bin
          else (ps.pytorch.override {cudaSupport = true;});
      in [
        torch
        ps.transformers
        ps.sentencepiece
        ps.accelerate
        ps.huggingface-hub
        ps.safetensors
        ps.ctranslate2 # Python bindings + ct2-transformers-converter
      ]);
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          pythonEnv
          pkgs.ctranslate2 # C++ libs/CLI built with CUDA (from overlay)
          pkgs.git
        ];

        NIXPKGS_ALLOW_UNFREE = "1";

        shellHook = ''
                      echo "--- CUDA/NLP sanity check ---"
                      python - <<'PY'
          import torch, ctranslate2 as c2, shutil
          print("torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
          if torch.cuda.is_available():
              print("GPU:", torch.cuda.get_device_name(0))
          print("ctranslate2:", getattr(c2, "__version__", "unknown"),
                "GPUs visible:", getattr(c2, "get_cuda_device_count", lambda: "n/a")())
          print("converter present:", shutil.which("ct2-transformers-converter") is not None)
          PY
                      echo
                      echo "Example:"
                      echo "  ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-ct2"
                      echo "  python - <<'PY'\nimport ctranslate2, transformers\nTGT='eng_Latn'; SRC='spa_Latn'\ntok=transformers.AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M',src_lang=SRC)\ntr=ctranslate2.Translator('nllb-ct2', device='cuda', compute_type='float16')\nsrc=tok.convert_ids_to_tokens(tok.encode('hola mundo'))\nres=tr.translate_batch([src], target_prefix=[[TGT]])\ntoks=res[0].hypotheses[0][1:]\nprint(tok.decode(tok.convert_tokens_to_ids(toks)))\nPY"
        '';
      };
    });
}
