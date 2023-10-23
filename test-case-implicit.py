import numpy as np

import slimplectic, orbit_util as orbit

G = 39.478758435  #(in AU^3/M_sun/yr^2))
M_Sun = 1.0  #(in solar masses)
rho = 2.0  #(in g/cm^3)
d = 5.0e-3  #(in cm)
beta = 0.0576906*(2.0/rho)*(1.0e-3/d)  #(dimensionless)
c = 63241.3  #(in AU/yr)
m = 1.

pr = slimplectic.GalerkinGaussLobatto('t', ['x', 'y'], ['vx', 'vy'])

# Define the conservative $L$ and nonconservative $K$ parts of the total Lagrangian $\Lambda$
# We take the dust particle to have unit mass.
L = 0.5*np.dot(pr.v, pr.v) + (1.0 - beta)*G*M_Sun/np.dot(pr.q, pr.q)**0.5
K = np.dot(pr.vp, pr.qm) + np.dot(pr.vp, pr.qp)*np.dot(pr.qp, pr.qm)/np.dot(pr.qp, pr.qp)
K *= -beta*G*M_Sun/c/np.dot(pr.qp, pr.qp)

# Discretize total Lagrangian using a 2nd order (r=0) implicit scheme
pr.discretize(L, K, 0, method='implicit')

# Specify time samples at which the numerical solution is to be given and provide initial data.

# We take the initial orbital parameters to be given by:
# a=1, e=0, i=0, omega=0, Omega=0, M=0
q0, v0 = orbit.Calc_Cartesian(1.0, 0.2, 0.0, 0.0, 0.0, 0.0, (1.0-beta)*G*M_Sun)
pi0 = v0  # Dust taken to have unit mass

# Time samples (in years)
t_end = 6000
dt = 0.01
t = np.arange(0, t_end+dt, dt)

# Discretize total Lagrangian using a 2nd order (r=0) implicit scheme
pr.discretize(L, K, 0, method='implicit')

# Now integrate the 2nd order slimplectic integrator
q_slim2, pi_slim2, v_slim2 = pr.integrate(q0[:2], pi0[:2], t, output_v=True)

# For a 4th order (r=1) implicit scheme we run
pr.discretize(L, K, 1, method='implicit')

# ...and then integrate to get the corresponding numerical solution
q_slim4, pi_slim4, v_slim4 = pr.integrate(q0[:2], pi0[:2], t, output_v=True)

pr.discretize(L, K, 2, method='implicit')  # 6th order is r=2
q_slim6, pi_slim6, v_slim6 = pr.integrate(q0[:2], pi0[:2], t, output_v=True)

data = np.vstack([
    q_slim2, pi_slim2, v_slim2,
    q_slim4, pi_slim4, v_slim4,
    q_slim6, pi_slim6, v_slim6
])

import hashlib
import sys
import datetime

output_file = f"test-case-outputs/{datetime.datetime.now().isoformat()}.csv"
# write numpy array to csv
np.savetxt(output_file, data, delimiter=",")


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
