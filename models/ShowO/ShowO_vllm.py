# Minimal Show-o wrapper with CLIP-ViT path only.
# Matches the Llava interface: __init__(model_path, args), process_messages(), generate_output(), generate_outputs()
# Dependencies expected in your repo:
#   models: Showo, CLIPVisionTower
#   training.prompting_utils: UniversalPrompting, create_attention_mask_for_mmu_vit
#   training.utils: image_transform (not used here), but keep if you plan to extend
#   transformers: AutoTokenizer, CLIPImageProcessor
#   llava.llava.conversation: conv_templates["phi1.5"]

import os
from typing import Dict, Any, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

# Show-o core (vendored under third_party/)
from third_party.showo.models import Showo
from third_party.showo.models import CLIPVisionTower
from third_party.showo.training.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_for_mmu_vit,
)

# LLaVA conversation templates (common path)

from third_party.showo.llava.llava import conversation as conversation_lib

# Use the same conversation template as the original script
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]

# Fixed system prompt text (we’ll compute its token length dynamically)
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

class ShowO:
    def __init__(self, model_path: str, args):
        """
        :param model_path: path or HF id for the Show-o checkpoint (used for both model and tokenizer).
        :param args: object with attributes: temperature, top_p, repetition_penalty, max_new_tokens
                     (these are accepted for interface parity; only max_new_tokens is used)
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer (left padding like original)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", padding_side="left")

        # UniversalPrompting with the special tokens Show-o expects
        self.uni = UniversalPrompting(
            self.tokenizer,
            max_text_len=getattr(args, "max_seq_length", 4096),
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100,
            cond_dropout_prob=getattr(args, "cond_dropout_prob", 0.0),
        )

        # Vision tower + processor (ViT-336 as in the original)
        self.vision_tower_name = "openai/clip-vit-large-patch14-336"
        self.vision_tower = CLIPVisionTower(self.vision_tower_name).to(self.device).eval()
        self.clip_proc = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        # Show-o model
        self.model = Showo.from_pretrained(model_path).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Sampling controls (keep interface but stay minimal)
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 1024))
        # Keep top_k=1 to mirror the original mmu demo’s greedy setting
        self.top_k = int(getattr(args, "top_k", 1))

    # ---------- helpers ----------
    def _ensure_pil(self, img: Union[str, Image.Image]) -> Image.Image:
        if isinstance(img, str):
            # path → PIL
            return Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError("image must be a str path or PIL.Image.Image")

    def _build_prompt_ids(self, question: str):
        # Conversation formatting (phi1.5 template), like the original script
        conv = conversation_lib.default_conversation.copy()
        # conv = llava_conv_templates["phi1.5"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt().strip()

        # Tokenize system + prompt
        input_ids_system = self.uni.text_tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding="longest").input_ids[0].to(self.device)
        sys_len = input_ids_system.shape[-1]  # dynamic system prompt length

        input_ids_prompt = self.uni.text_tokenizer(prompt_str, return_tensors="pt", padding="longest").input_ids[0].to(self.device)

        # Compose: <|mmu|> + SYSTEM + <|soi|> + <|eoi|> + user prompt
        ids = torch.cat([
            torch.tensor([self.uni.sptids_dict['<|mmu|>']], device=self.device, dtype=torch.long),
            input_ids_system,
            torch.tensor([self.uni.sptids_dict['<|soi|>']], device=self.device, dtype=torch.long),
            torch.tensor([self.uni.sptids_dict['<|eoi|>']], device=self.device, dtype=torch.long),
            input_ids_prompt,
        ], dim=0).unsqueeze(0)  # shape: (1, L)
        return ids, sys_len

    def _encode_image_embeddings(self, pil_img: Image.Image):
        pixel_values = self.clip_proc.preprocess(pil_img, return_tensors="pt")["pixel_values"].to(self.device)  # (1, 3, 336, 336)
        with torch.no_grad():
            img_emb = self.vision_tower(pixel_values)                    # (1, T_v, D_v)
            img_emb = self.model.mm_projector(img_emb)                   # (1, T_v, D_llm)
        return img_emb

    def _prepare_inputs(self, messages: Dict[str, Any]):
        """Build input_embeddings and attention_mask for Show-o (ViT path)."""
        # pull text
        prompt = messages["prompt"]

        # pick the image: prefer 'image', else first of 'images'
        img: Optional[Union[str, Image.Image]] = None
        if "image" in messages:
            img = messages["image"]
        elif "images" in messages and messages["images"]:
            img = messages["images"][0]
        # Allow text-only, but Show-o MMU expects an image; raise if missing
        if img is None:
            raise ValueError("Show-o (ViT path) requires an 'image' in messages.")

        pil_img = self._ensure_pil(img)

        # Build tokens for (system + chat prompt)
        input_ids_llava, sys_len = self._build_prompt_ids(prompt)

        # Embeddings
        with torch.no_grad():
            text_emb = self.model.showo.model.embed_tokens(input_ids_llava)  # (1, L, D)
            img_emb = self._encode_image_embeddings(pil_img)                 # (1, T_v, D)

            # Insert image embeddings after <|mmu|> + SYSTEM + <|soi|>
            # original code: part1 = [: 2 + SYSTEM_PROMPT_LEN], part2 = [rest]
            # here we use dynamic sys_len instead of hard-coded 28
            split_idx = 2 + sys_len  # 1 (<|mmu|>) + sys_len + 1 (<|soi|>)
            part1 = text_emb[:, :split_idx, :]
            part2 = text_emb[:, split_idx:, :]
            input_embeddings = torch.cat((part1, img_emb, part2), dim=1)

            # Build attention mask for ViT path
            attention_mask = create_attention_mask_for_mmu_vit(
                input_embeddings, system_prompt_len=sys_len
            )[0].unsqueeze(0)

        return input_embeddings, attention_mask

    # ---------- public interface (mirrors Llava) ----------
    def process_messages(self, messages: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict of tensors we’ll feed into generate()."""
        input_embeddings, attention_mask = self._prepare_inputs(messages)
        return {
            "input_embeddings": input_embeddings,
            "attention_mask": attention_mask,
        }

    def generate_output(self, messages: Dict[str, Any]) -> str:
        batch = self.process_messages(messages)
        with torch.no_grad():
            cont_toks_list = self.model.mmu_generate(
                input_embeddings=batch["input_embeddings"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                top_k=self.top_k,
                eot_token=self.tokenizer.eos_token_id,  # simple EOS; no extra handling
            )
        # cont_toks_list: list[tensor] → (1, L_gen)
        tokens = torch.stack(cont_toks_list).squeeze()[None]
        text = self.uni.text_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return text

    def generate_outputs(self, messages_list: List[Dict[str, Any]]) -> List[str]:
        results: List[str] = []
        for messages in messages_list:
            results.append(self.generate_output(messages))
        return results
