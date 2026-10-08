import os
import numpy as np
import matplotlib.pyplot as plt

K = 2          # number of clusters
SEED = None    # fixed number = same result every run | None = random every run
MAX_ITER = 100

# ---------- Input: read images ----------
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_set")
files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
names = [f.split("_")[1].split(".")[0] for f in files]   # 6752300496_Max.jpg -> Max

imgs = []
X = []
for f in files:
    x = plt.imread(os.path.join(folder, f))
    if x.dtype != np.uint8:    # png is read as 0-1, scale to 0-255 like jpg
        x = x * 255
    imgs.append(x)
    X.append(x.flatten())
X = np.array(X, dtype=float)
N = len(X)
assert 1 <= K <= N, f"K must be between 1 and {N}"

# ---------- K-means ----------
np.random.seed(SEED)
C = X[np.random.permutation(N)[:K]]
labels = None
for it in range(1, MAX_ITER + 1):
    D = np.zeros((N, K))
    for i in range(K):
        D[:, i] = np.sum((X - C[i]) ** 2, axis=1)
    new_labels = np.argmin(D, axis=1)

    size = np.bincount(new_labels, minlength=K).tolist()
    if labels is None:
        print(f"Iter {it}: cluster sizes = {size}")
    else:
        print(f"Iter {it}: cluster sizes = {size}, changed = {np.sum(new_labels != labels)}")
    if labels is not None and np.array_equal(new_labels, labels):
        print("Stop: labels did not change")
        break
    labels = new_labels

    for i in range(K):
        if np.any(labels == i):
            C[i] = np.mean(X[labels == i], axis=0)
else:
    print(f"Stop: reached {MAX_ITER} iterations but labels are still changing")

# ---------- Output: who is in each cluster ----------
print()
for i in range(K):
    members = [names[j] for j in range(N) if labels[j] == i]
    print(f"Cluster {i + 1}: " + (", ".join(members) if members else "(empty)"))

# ---------- Images of each cluster ----------
cols = max(np.bincount(labels, minlength=K).max(), 1)
fig, axes = plt.subplots(K, cols, figsize=(cols * 1.4 + 1, K * 2), squeeze=False)
for ax in axes.flatten():
    ax.axis("off")
for i in range(K):
    axes[i, 0].text(-0.15, 0.5, f"Cluster {i + 1}", transform=axes[i, 0].transAxes,
                    ha="right", va="center", fontsize=10)
    for c, j in enumerate(np.where(labels == i)[0]):
        axes[i, c].imshow(imgs[j], cmap="gray", vmin=0, vmax=255)
        axes[i, c].set_title(names[j], fontsize=8)
plt.subplots_adjust(left=0.1)
plt.show()
