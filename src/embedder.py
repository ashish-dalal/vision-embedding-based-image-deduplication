import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class ClipEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model.eval()

    @torch.no_grad()
    def encode(self, pil_images, batch_size=32):
        vecs = []
        for i in tqdm(range(0, len(pil_images), batch_size), desc="Embedding"):
            batch = pil_images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.cpu().numpy())
        return np.concatenate(vecs, axis=0)