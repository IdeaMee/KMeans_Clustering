import os
import numpy as np
import matplotlib.pyplot as plt

K = 3  # clusters

# ---------- Input: read images ----------
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_set")  # Data_set folder
files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
names = [f.split("_")[1].split(".")[0] for f in files]

imgs = [plt.imread(os.path.join(folder, f)) for f in files]
X = np.array([x.flatten() for x in imgs], dtype=float)

# ---------- K-means ----------
pick = np.random.permutation(len(X))[:K]
C = X[pick]
