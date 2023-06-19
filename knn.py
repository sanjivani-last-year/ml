#Normal KNN
#matrix_matrix_multiplication
import random
import numpy as np
import time

N = 100
A = np.random.randint(0, N, size=(N, N))
B = np.random.randint(0, N, size=(N, N))
C = np.zeros((N, N), dtype=int)

start = time.time()
for i in range(N):
    for j in range(N):
        C[i][j] = 0
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]
duration = time.time() - start
print("Serial Multiply Matrix and matrix:", duration)

start = time.time()
C = np.dot(A, B)
duration = time.time() - start
print("Parallel Multiply Matrix and matrix:", duration)

#matrix_vector_multiplication
import random
import numpy as np
import time
M = 700
# Multiply Vector and matrix
A = np.random.randint(0, M, size=(M, M))
a = np.random.randint(0, M, size=M)
d = np.zeros(M, dtype=int)
# serial
start = time.time()
for i in range(M):
    sum = 0
    for j in range(M):
        sum += A[i][j] * a[j]
    d[i] = sum
duration = time.time() - start
print("Serial Multiply Vector and matrix:", duration)

# parallel
start = time.time()
d = np.dot(A, a)
duration = time.time() - start
print("Parallel Multiply Vector and matrix:", duration)

#vector_vector_addition
N = 10000000

# Addition of two vectors
a = [random.randint(0, N) for _ in range(N)]
b = [random.randint(0, N) for _ in range(N)]
c = np.zeros(N, dtype=int)

start = time.time()
for i in range(N):
    c[i] = b[i] + a[i]
duration = time.time() - start
print("Serial vector addition:", duration)

start = time.time()
c = np.add(a, b)
duration = time.time() - start
print("Parallel vector addition:", duration)

#Parallel KNN
from collections import Counter
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances

class KNN:
    def __init__(self, k=3, n_jobs=1):
        self.k = k
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        dists = pairwise_distances(X, self.X_train, n_jobs=self.n_jobs)
        indices = np.argsort(dists, axis=1)[:, :self.k]
        y_pred = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)(indices[i]) for i in range(X.shape[0]))
        return np.array(y_pred)

    def _predict(self, indices):
        k_neighbor_labels = self.y_train[indices]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    k = 3
    n_jobs = 4  # Number of parallel jobs

    clf = KNN(k=k, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy:", accuracy(y_test, predictions))
