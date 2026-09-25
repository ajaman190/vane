"""Optional frozen Hugging Face encoder. Not imported by tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class FrozenEncoder:
    """A published encoder whose weights do not train."""

    repo: str
    tokenizer: object
    model: nn.Module
    dim: int

    @classmethod
    def load(cls, repo: str, device: torch.device | str) -> "FrozenEncoder":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Loading a backbone requires transformers. "
                "Install it in the train environment."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo)
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        dim = int(getattr(model.config, "hidden_size", 768))
        return cls(repo=repo, tokenizer=tokenizer, model=model, dim=dim)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], max_length: int) -> tuple[Tensor, Tensor]:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].to(dtype=torch.bool)
        return hidden, mask


@dataclass
class FrozenVision:
    """Frozen vision tower. Returns patch states, not a caption."""

    repo: str
    processor: object
    model: nn.Module
    dim: int

    @classmethod
    def load(cls, repo: str, device: torch.device | str) -> "FrozenVision":
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError("Loading a vision backbone requires transformers.") from exc
        processor = AutoProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo)
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        dim = int(getattr(getattr(model.config, "vision_config", model.config), "hidden_size", 768))
        return cls(repo=repo, processor=processor, model=model, dim=dim)

    @torch.no_grad()
    def embed_images(self, images: list[object]) -> Tensor:
        import io

        from PIL import Image

        pil = []
        for image in images:
            if image is None:
                pil.append(Image.new("RGB", (224, 224), color=0))
            elif isinstance(image, bytes):
                pil.append(Image.open(io.BytesIO(image)).convert("RGB"))
            else:
                pil.append(image.convert("RGB") if hasattr(image, "convert") else image)
        inputs = self.processor(images=pil, return_tensors="pt")
        pixel = inputs["pixel_values"].to(next(self.model.parameters()).device)
        vision = getattr(self.model, "vision_model", None)
        if vision is None:
            hidden = self.model.get_image_features(pixel_values=pixel)
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(1)
            return hidden
        return vision(pixel_values=pixel).last_hidden_state
