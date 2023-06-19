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
