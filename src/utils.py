from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder):
    folder = Path(folder)
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return None