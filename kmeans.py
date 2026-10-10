import os
import numpy as np
import matplotlib.pyplot as plt

K = 2          # number of clusters
SEED = None    # fixed number = same result every run | None = random every run

# ---------- Input: read images ----------
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_set")
files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
names = [f.split("_")[1].split(".")[0] for f in files]   # 6752300496_Max.jpg -> Max

imgs = []
for f in files:
    x = plt.imread(os.path.join(folder, f))
    if x.dtype != np.uint8:    # png is read as 0-1, scale to 0-255 like jpg
        x = x * 255
    imgs.append(x)
X = np.array([x.flatten() for x in imgs], dtype=float)

# ---------- K-means (Manhattan distance) ----------
np.random.seed(SEED)
C = X[np.random.permutation(len(X))[:K]]
for it in range(1, 21):
    D = np.zeros((len(X), K))
    for k in range(K):
        D[:, k] = np.sum(np.abs(X - C[k]), axis=1)
    idx = np.argmin(D, axis=1)
    print(f"Iter {it}: cluster sizes = {np.bincount(idx, minlength=K).tolist()}")
    for k in range(K):
        if np.any(idx == k):
            C[k] = np.mean(X[idx == k], axis=0)

# ---------- Output: who is in each cluster ----------
print()
for k in range(K):
    members = [names[j] for j in range(len(X)) if idx[j] == k]
    print(f"Cluster {k + 1}: " + (", ".join(members) if members else "(empty)"))
