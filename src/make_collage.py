from pathlib import Path
from PIL import Image

def make_collages(out_dir="visioncache_output/clusters", cols=5, thumb=128):
    root = Path(out_dir)
    clusters = sorted(root.glob("cluster_*"))
    if not clusters:
        print("No clusters found.")
        return
    rows = len(clusters)
    print(f"Generating {rows} rows of collages...")

    row_imgs = []
    for c in clusters:
        imgs = list(c.glob("*.jpg"))
        imgs = imgs[:cols]  # limit per row
        thumbs = [Image.open(p).convert("RGB").resize((thumb, thumb)) for p in imgs]
        row_width = thumb * len(thumbs)
        row_img = Image.new("RGB", (row_width, thumb))
        for i, im in enumerate(thumbs):
            row_img.paste(im, (i * thumb, 0))
        row_imgs.append(row_img)

    # combine rows vertically
    total_h = thumb * len(row_imgs)
    max_w = max(r.width for r in row_imgs)
    canvas = Image.new("RGB", (max_w, total_h), (255,255,255))
    y = 0
    for r in row_imgs:
        canvas.paste(r, (0, y))
        y += thumb
    canvas.save("visioncache_output/collages.png")
    print("Saved to visioncache_output/collages.png")

if __name__ == "__main__":
    make_collages()
