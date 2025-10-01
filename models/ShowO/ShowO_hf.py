# models/ShowO/ShowO_hf.py
# Wrapper to use Show-o inside MedEvalKit without clashing with MedEvalKit.models
# - Loads Show-o repo modules by absolute path under unique names
# - Uses base LLM tokenizer (as in Show-o training)
# - Supports simple text gen and basic VQA/MMU (image + question)

import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from transformers import AutoTokenizer


# ========= CHANGE THIS to your local Show-o repo root =========
SHOWO_ROOT = "/home/azureuser/disk/Show-o"
# =============================================================


def _load_package(pkg_name: str, pkg_dir: str):
    """
    Load a package by directory so its relative imports (.) work.
    (Needed because Show-o has a top-level 'models' package that clashes with MedEvalKit.models)
    """
    init_py = os.path.join(pkg_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        pkg_name, init_py, submodule_search_locations=[pkg_dir]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module  # allow relative imports within the package
    assert spec.loader is not None, f"Cannot load package at {pkg_dir}"
    spec.loader.exec_module(module)
    return module


def _load_module(mod_name: str, file_path: str):
    """Load a single-file module by absolute path under a unique name."""
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    assert spec.loader is not None, f"Cannot load module at {file_path}"
    spec.loader.exec_module(module)
    return module


# ---- Load Show-o code by path (avoid 'models' name collision) ----
showo_models = _load_package("showo_models", os.path.join(SHOWO_ROOT, "models"))
showo_prompt = _load_module("showo_prompt", os.path.join(SHOWO_ROOT, "training", "prompting_utils.py"))

# Pull required symbols from the Show-o repo
Showo = getattr(showo_models, "Showo")
MAGVITv2 = getattr(showo_models, "MAGVITv2")

# get_mask_chedule may be exposed either in models or prompting_utils depending on commit
get_mask_chedule = getattr(showo_models, "get_mask_chedule", None)
if get_mask_chedule is None:
    get_mask_chedule = getattr(showo_prompt, "get_mask_chedule")

UniversalPrompting = showo_prompt.UniversalPrompting
create_attention_mask_predict_next = showo_prompt.create_attention_mask_predict_next
create_attention_mask_for_mmu = showo_prompt.create_attention_mask_for_mmu


class ShowO:
    """
    Minimal Show-o wrapper for MedEvalKit:
      - Text generation via Show-o LLM head
      - VQA/MMU (image + question) via MAGVITv2 -> VQ tokens + UniversalPrompting

    Required:
      - model_path: folder with Show-o checkpoint saved via save_pretrained(...)
      - tokenizer_path: base LLM tokenizer path (e.g., Llama/Qwen tokenizer used during Show-o training)
                        pass via args.tokenizer_path or env SHOWO_TOKENIZER, or embedded in model_path/config.json
    """

    def __init__(
        self,
        model_path: str,
        args,
        tokenizer_path: Optional[str] = None,
        vq_pretrained_name: Optional[str] = None,   # use if your MAGVITv2 exposes a named 'from_pretrained'
        vq_state_dict_path: Optional[str] = None,   # or provide a local state dict checkpoint
        use_bf16_if_available: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- define dtype FIRST (before any .from_pretrained that uses it) ---
        self.dtype = (
            torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        
        # (then resolve tokenizer_path, load tokenizer, VQ, etc.)
        tok_path = self._resolve_tokenizer_path(model_path, tokenizer_path, args)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, padding_side="left", use_fast=False)
        
        self.vq_model = self._load_vq_model(vq_pretrained_name, vq_state_dict_path)
        self.vq_model.to(self.device).eval().requires_grad_(False)
        
        # --- load Show-o without meta init, using the dtype you set above ---
        self.model = Showo.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,   # avoid init_empty_weights/meta path
            device_map=None,           # do NOT use "auto" here
            torch_dtype=self.dtype
        ).eval()
        self.model.to(self.device)

        # --- Universal prompting & mask schedule (mirrors Show-o training) ---
        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=2048,  # adjust if you have this in your saved config
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
            ),
            ignore_id=-100,
            cond_dropout_prob=0.0,
        )
        self.mask_schedule = get_mask_chedule("cosine")

        # dtype preference for FA2 / inference speed
        self.dtype = (
            torch.bfloat16 if (use_bf16_if_available and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )

        # cache some config values if present
        cfg = getattr(self.model, "config", None)
        self.num_vq_tokens = getattr(cfg, "num_vq_tokens", 256)

    # ---------- Public API expected by MedEvalKit ----------

    @torch.no_grad()
    def __call__(self, prompt: str, **gen_kw):
        """Text-only generation shortcut."""
        return self.generate_text(prompt, **gen_kw)

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.8) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, temperature=temperature)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def vqa(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> str:
        """
        Basic VQA/MMU pipeline:
          PIL/image path -> [-1,1] tensor -> MAGVITv2.get_code() -> shift by text tokenizer size
          -> UniversalPrompting (mmu) -> attention mask -> generate answer tokens.
        """
        img = self._load_pil(image)
        pixel = self._pil_to_tensor(img).unsqueeze(0).to(self.device)  # [1,3,H,W] in [-1,1]

        # Image to discrete tokens (codebook ids), then shift by |text_tokenizer| (training convention)
        image_tokens = self.vq_model.get_code(pixel)                   # [1, num_vq_tokens]
        image_tokens = image_tokens + len(self.uni_prompting.text_tokenizer)

        # Build MMU input ids (labels unused for inference)
        input_ids_mmu, _, _ = self.uni_prompting((image_tokens, [question]), 'mmu')
        input_ids_mmu = input_ids_mmu.to(self.device)

        # Attention mask as in training
        eoi_id = int(self.uni_prompting.sptids_dict['<|eoi|>'])
        attn_mask = create_attention_mask_for_mmu(input_ids_mmu, eoi_id=eoi_id)

        with torch.autocast("cuda", dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            out_ids = self.model.generate(
                input_ids=input_ids_mmu,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # ---------- helpers ----------

    def _resolve_tokenizer_path(self, model_path: str, tokenizer_path: Optional[str], args) -> str:
        # 1) explicit arg
        if tokenizer_path:
            return tokenizer_path

        # 2) try model_path/config.json saved during training
        cfg_file = Path(model_path) / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                tok = (
                    cfg.get("showo", {}).get("llm_model_path")
                    or cfg.get("model", {}).get("showo", {}).get("llm_model_path")
                    or cfg.get("llm_model_path")
                )
                if tok:
                    return tok
            except Exception:
                pass

        # 3) CLI arg (if you exposed --tokenizer_path in MedEvalKit)
        if hasattr(args, "tokenizer_path") and getattr(args, "tokenizer_path", None):
            return args.tokenizer_path

        # 4) environment variable
        tok = os.environ.get("SHOWO_TOKENIZER")
        if tok:
            return tok

        raise ValueError(
            "Show-o needs a base LLM tokenizer path.\n"
            "Provide --tokenizer_path, or set SHOWO_TOKENIZER, or ensure model_path/config.json contains llm_model_path."
        )

    def _load_vq_model(self, vq_pretrained_name: Optional[str], vq_state_dict_path: Optional[str]):
        # If you have a local VQ state dict (from training), prefer it
        if vq_state_dict_path:
            vq = MAGVITv2()
            state = torch.load(vq_state_dict_path, map_location="cpu")["model"]
            vq.load_state_dict(state)
            return vq
        # If Show-o's MAGVITv2 exposes a named 'from_pretrained', use it
        if hasattr(MAGVITv2, "from_pretrained") and vq_pretrained_name:
            return MAGVITv2.from_pretrained(vq_pretrained_name)
        # Fallback to a plain init (assumes internal weights are bundled or not needed)
        return MAGVITv2()

    @staticmethod
    def _load_pil(x: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        return Image.open(x).convert("RGB")

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """Match training’s normalization: float32 CHW in [-1, 1]."""
        import numpy as np
        arr = (np.array(img).astype("float32") / 255.0) * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
