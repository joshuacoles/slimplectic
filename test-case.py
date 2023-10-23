import sys

import slimplectic
import numpy as np
import datetime

m = 1.0
k = 1.0
lamda = 1e-4 * np.sqrt(m * k)

dho = slimplectic.GalerkinGaussLobatto('t', ['q'], ['v'])

L = 0.5 * m * np.dot(dho.v, dho.v) - 0.5 * k * np.dot(dho.q, dho.q)
K = -lamda * np.dot(dho.vp, dho.qm)

dho.discretize(L, K, 0, method='explicit', verbose=True)

# Specify time samples at which the numerical solution is to be given initial data

# Time samples
dt = 0.1 * np.sqrt(m / k)
tmax = 10000 * np.sqrt(m / k)
t = dt * np.arange(0, int(tmax / dt) + 1)

# Initial data (at t=0)
q0 = [1.]
pi0 = [0.25 * dt * k]

# The initial condition for pi0 is chosen because the 2nd order slimplectic method
# has $\pi$ actually evaluated at the mid-step, and it needs corrections to that effect.
# Otherwise, the phase is off and the energy has a constant offset.

q_slim2, pi_slim2, v_slim2 = dho.integrate(q0, pi0, t, output_v=True)

# For a 4th order (r=1) explicit scheme we run
dho.discretize(L, K, 1, method='explicit')
q_slim4, pi_slim4, v_slim4 = dho.integrate(q0, pi0, t, output_v=True)

data = np.vstack([
    q_slim2, pi_slim2, v_slim2,
    q_slim4, pi_slim4, v_slim4
])

output_file = f"test-case-outputs/{datetime.datetime.now().isoformat()}.csv"
# write numpy array to csv
np.savetxt(output_file, data, delimiter=",")

import hashlib

def hash_file(file_path):
    # Create a hash object
    hasher = hashlib.sha256()

    # Read the file in chunks and update the hash object
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            hasher.update(chunk)

    # Return the hexadecimal representation of the hash
    return hasher.hexdigest()

def compare_files(file1, file2):
    # Hash the first file
    hash1 = hash_file(file1)

    # Hash the second file
    hash2 = hash_file(file2)

    return hash1 == hash2

correct = compare_files(
    sys.argv[1],
    output_file
)


if correct:
    print("Working")
else:
    print(f"Failed see {output_file}")
    exit(1)
