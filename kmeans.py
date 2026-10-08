import os
import numpy as np
import matplotlib.pyplot as plt

K = 4          # number of clusters
SEED = 0    # fixed number = same result every run | None = random every run
MAX_ITER = 100

# ---------- Input: read images ----------
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_set")
files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
names = [f.split("_")[1].split(".")[0] for f in files]   # 6752300496_Max.jpg -> Max

imgs = [plt.imread(os.path.join(folder, f)) for f in files]   # N images, each (180, 140)
X = np.array([x.flatten() for x in imgs], dtype=float)   # (N, 25200)

# ---------- K-means (Manhattan distance) ----------
np.random.seed(SEED)
C = X[np.random.permutation(len(X))[:K]]                 # (K, 25200)
it = 0
while True:
    it += 1
    D = np.zeros((len(X), K))                            # (N, K)
    for k in range(K):
        D[:, k] = np.sum(np.abs(X - C[k]), axis=1)       # (N,)
    idx = np.argmin(D, axis=1)                           # (N,)
    print(f"Iter {it}: cluster sizes = {np.bincount(idx, minlength=K).tolist()}")
    Cold = C.copy()
    for k in range(K):
        if np.any(idx == k):
            C[k] = np.mean(X[idx == k], axis=0)          # (25200,)
    if np.sum(np.abs(Cold - C)) == 0:
        print("finished")
        break
    if it == MAX_ITER:
        print(f"stopped: reached {MAX_ITER} iterations")
        break

# ---------- Output: who is in each cluster ----------
print()
for k in range(K):
    members = [names[j] for j in range(len(X)) if idx[j] == k]
    print(f"Cluster {k + 1}: " + (", ".join(members) if members else "(empty)"))

# ---------- Images of each cluster ----------
cols = max(np.bincount(idx, minlength=K).max(), 1)
fig, axes = plt.subplots(K, cols, figsize=(cols * 1.4 + 1, K * 2), squeeze=False)
for ax in axes.flatten():
    ax.axis("off")
for k in range(K):
    axes[k, 0].text(-0.15, 0.5, f"Cluster {k + 1}", transform=axes[k, 0].transAxes,
                    ha="right", va="center", fontsize=10)
    for c, j in enumerate(np.where(idx == k)[0]):
        axes[k, c].imshow(imgs[j], cmap="gray", vmin=0, vmax=255)
        axes[k, c].set_title(names[j], fontsize=8)
plt.subplots_adjust(left=0.1)
plt.show()
