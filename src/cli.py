import argparse
import os
import faiss, numpy as np, json
from pathlib import Path
from utils import list_images, load_image
from embedder import ClipEmbedder
import numpy as np, json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="resource_tracker")

def index_cmd(args):
    images = list_images(args.images_dir)
    print(f"Found {len(images)} images.")

    pil_list = [load_image(p) for p in images if load_image(p)]
    
    model = ClipEmbedder()
    vecs = model.encode(pil_list)
    
    out_dir = Path("visioncache_output")
    out_dir.mkdir(exist_ok=True)
    
    np.save(out_dir / "embeddings.npy", vecs)
    (out_dir / "files.json").write_text(json.dumps([str(p) for p in images], indent=2))
    print("Embeddings saved to visioncache_output/")

def cluster_cmd(args):
    from sklearn.cluster import AgglomerativeClustering

    out = Path(args.out_dir)
    vecs = np.load(out / "embeddings.npy").astype("float32")
    files = json.loads((out / "files.json").read_text())

    print("Clustering...")
    sim_thresh = 0.86  # for stricter / looser grouping
    dist_thresh = 1 - sim_thresh

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh
    )
    labels = clustering.fit_predict(vecs)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(files[i])

    cluster_root = out / "clusters"
    cluster_root.mkdir(exist_ok=True)

    for cid, paths in clusters.items():
        if len(paths) < 2:  # skip singletons
            continue
        cdir = cluster_root / f"cluster_{cid:04d}"
        cdir.mkdir(exist_ok=True)
        for p in paths:
            name = Path(p).name
            os.link(p, cdir / name) if not (cdir / name).exists() else None

    print(f"Clusters written to {cluster_root}")

def find_cmd(args):
    out = Path(args.out_dir)
    index = faiss.read_index(str(out / "faiss.index"))
    vecs = np.load(out / "embeddings.npy").astype("float32")
    files = json.loads((out / "files.json").read_text())

    from embedder import ClipEmbedder
    from PIL import Image

    model = ClipEmbedder()
    img = Image.open(args.image).convert("RGB")
    q = model.encode([img]).astype("float32")

    D, I = index.search(q, 5)
    print("\nTop-5 similar images:")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        print(f"{rank}. {files[idx]}  (cos={dist:.4f})")

def main():
    parser = argparse.ArgumentParser(prog="visioncache", description="VisionCache CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p1 = sub.add_parser("index", help="Embed and index images")
    p1.add_argument("--images_dir", required=True)
    p1.set_defaults(func=index_cmd)

    # cluster
    p2 = sub.add_parser("cluster", help="Cluster indexed images")
    p2.add_argument("--out_dir", required=True)
    p2.set_defaults(func=cluster_cmd)

    # find
    p3 = sub.add_parser("find", help="Find near-duplicates for a query image")
    p3.add_argument("--out_dir", required=True)
    p3.add_argument("--image", required=True)
    p3.set_defaults(func=find_cmd)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()