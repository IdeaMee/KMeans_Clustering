import os
import numpy as np
import matplotlib.pyplot as plt

K = 2          # number of clusters
SEED = None    # fixed number = same result every run | None = random every run

# ---------- Input: read images ----------
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_set")
files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
names = [f.split("_")[1].split(".")[0] for f in files]   # 6752300496_Max.jpg -> Max

imgs = [plt.imread(os.path.join(folder, f)) for f in files]
X = np.array([x.flatten() for x in imgs], dtype=float)

# ---------- K-means ----------
np.random.seed(SEED)
C = X[np.random.permutation(len(X))[:K]]
for it in range(20):
    D = np.zeros((len(X), K))
    for k in range(K):
        D[:, k] = np.sum((X - C[k]) ** 2, axis=1)
    idx = np.argmin(D, axis=1)
    for k in range(K):
        if np.any(idx == k):
            C[k] = np.mean(X[idx == k], axis=0)
print(idx)
