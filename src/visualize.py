import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def plot_and_save(vecs, labels, name):
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 7), dpi=150)

    palette = sns.color_palette("husl", len(set(labels)))
    plt.scatter(
        vecs[:, 0],
        vecs[:, 1],
        c=[palette[l % len(palette)] for l in labels],
        s=80,
        edgecolors="white",
        linewidth=0.5,
        alpha=0.9
    )
    plt.title(name.upper(), fontsize=14, weight="bold", pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"visioncache_output/{name}.png", bbox_inches="tight")
    plt.close()


def main():
    X = np.load("visioncache_output/embeddings.npy")
    labels = np.loadtxt("visioncache_output/labels.txt", dtype=int) if \
              (Path("visioncache_output/labels.txt").exists()) else np.zeros(len(X))
    labels = labels.astype(int)

    print("PCA...")
    pca = PCA(n_components=2).fit_transform(X)
    plot_and_save(pca, labels, "pca")

    print("t-SNE...")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=3).fit_transform(X)
    plot_and_save(tsne, labels, "tsne")

    print("UMAP...")
    umap_emb = umap.UMAP(n_components=2).fit_transform(X)
    plot_and_save(umap_emb, labels, "umap")

if __name__ == "__main__":
    from pathlib import Path
    main()