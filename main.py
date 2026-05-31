import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import classifier

# Loading Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
irisDataFrame = pd.read_csv(url, header=None)

# --- PART A: Iris-setosa vs others ---
# Using first two dimensions (2D)
X_2d = irisDataFrame.iloc[:, :2].to_numpy()
label_names = irisDataFrame.iloc[:, 4].to_numpy()
labels_setosa = np.where(label_names == "Iris-setosa", 1, 0)

# Train LLS for 2D
alpha_2d, X_b_2d = classifier.run_lls(X_2d, labels_setosa)

# Plotting 2D
plt.title("Identifying Iris-setosa on the underlying Pandas-dataset (using 2 Dimensions).")
plt.scatter(X_2d[labels_setosa==1, 0], X_2d[labels_setosa==1, 1], label="Iris-setosa")
plt.scatter(X_2d[labels_setosa==0, 0], X_2d[labels_setosa==0, 1], label="Not Iris-setosa")
x1_vals = np.linspace(4, 8, 1000)
# Formula for the hyperplane
x2_vals = (0.5 - alpha_2d[0] - alpha_2d[1] * x1_vals) / alpha_2d[2]
plt.plot(x1_vals, x2_vals, color="black", label="Hyperplane")
plt.legend()
plt.show()

# Using all four dimensions (4D)
X_4d = irisDataFrame.iloc[:, :4].to_numpy()
alpha_4d, X_b_4d = classifier.run_lls(X_4d, labels_setosa)
c, acc = classifier.compute_confusion_matrix(X_b_4d, labels_setosa, alpha_4d)
print(f"Confusion Matrix (4D Setosa):\n{c}")
print(f"Accuracy (4D): {acc}")

print("\nHere we see that the confusion matrix is a diagonal matrix. Hence we know that in R^4, there is a hyperplane calculated by LLS, which can perfectly separate the set of Iris Setosa and not Iris Setosa. So we can clearly tell about our training data, which flowers are Setosa and which are not.")

# --- PART B: Iris-versicolor vs others ---
# Using first two dimensions (2D)
labels_versicolor = np.where(label_names == "Iris-versicolor", 1, 0)

# Train LLS for 2D (Versicolor)
alpha_v_2d, X_b_v_2d = classifier.run_lls(X_2d, labels_versicolor)

# Plotting 2D (Versicolor)
plt.title("Identifying Iris-versicolor on the underlying Pandas-dataset (using 2 Dimensions).")
plt.scatter(X_2d[labels_versicolor==1, 0], X_2d[labels_versicolor==1, 1], label="Iris-versicolor")
plt.scatter(X_2d[labels_versicolor==0, 0], X_2d[labels_versicolor==0, 1], label="Not Iris-versicolor")

x1_vals = np.linspace(4, 8, 1000)
x2_vals = (0.5 - alpha_v_2d[0] - alpha_v_2d[1] * x1_vals) / alpha_v_2d[2]
plt.plot(x1_vals, x2_vals, color="black", linewidth=2, label="Hyperplane")
plt.legend()
plt.title("Separation: Iris-versicolor vs Others (2D)")
plt.show()

# Using all four dimensions (4D)
alpha_v_4d, X_b_v_4d = classifier.run_lls(X_4d, labels_versicolor)
c_v, acc_v = classifier.compute_confusion_matrix(X_b_v_4d, labels_versicolor, alpha_v_4d)

print(f"Confusion Matrix (4D Versicolor):\n{c_v}")
print(f"Accuracy (4D): {acc_v}")

print("\nObservation: Here we see, that it is much harder for our algorithm to differentiate between Iris-versicolor and not Iris-versicolor. The reason for that is (as seen in the 2d plot) that it is impossible to seprate the two clusters. Unlike the first time, where the clusters were pretty much on their own")