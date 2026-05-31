import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


# -----------------------------
# 1. Define Hamiltonian
# -----------------------------

J = 1.0
g = 0.5

hamiltonian = SparsePauliOp.from_list([
    ("ZZ", J),
    ("XI", g),
    ("IX", g),
])

print("Hamiltonian:")
print(hamiltonian)


# -----------------------------
# 2. Exact ground energy
# -----------------------------

H_matrix = hamiltonian.to_matrix()
eigenvalues = np.linalg.eigvalsh(H_matrix)

E0_exact = eigenvalues[0]

print("\nExact eigenvalues:", eigenvalues.round(6))
print(f"Exact ground state energy E0 = {E0_exact:.6f}")


# -----------------------------
# 3. Build ansatz circuit
# -----------------------------

def build_ansatz(n_params: int = 6) -> QuantumCircuit:
    theta = ParameterVector("theta", n_params)
    qc = QuantumCircuit(2)

    qc.ry(theta[0], 0)
    qc.rz(theta[1], 0)

    qc.ry(theta[2], 1)
    qc.rz(theta[3], 1)

    qc.cx(1, 0)

    qc.rz(theta[4], 1)
    qc.ry(theta[5], 1)

    return qc


ansatz = build_ansatz()

print(f"\nAnsatz has {ansatz.num_parameters} parameters")
print(f"Ansatz depth: {ansatz.depth()}")
print(ansatz.draw())


# -----------------------------
# 4. Set up Estimator
# -----------------------------

estimator = AerEstimator()
estimator.options.default_shots = 8192


# -----------------------------
# 5. Test one random energy
# -----------------------------

test_params = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)

pub = (ansatz, hamiltonian, test_params)
job = estimator.run([pub])
result = job.result()

test_energy = result[0].data.evs

print("\nRandom test parameters:", test_params.round(3))
print(f"Energy at test parameters: {test_energy:.6f}")
print(f"Exact ground energy:        {E0_exact:.6f}")


# -----------------------------
# 6. VQE optimization
# -----------------------------

energy_history = []
param_history = []


def cost_function(params):
    pub = (ansatz, hamiltonian, params)
    job = estimator.run([pub])
    energy = job.result()[0].data.evs

    energy_history.append(float(energy))
    param_history.append(params.copy())

    return float(energy)


np.random.seed(42)
theta_init = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

print("\nInitial parameters:", theta_init.round(3))
print(f"Initial energy:      {cost_function(theta_init):.6f}")
print(f"Exact ground energy: {E0_exact:.6f}")

print("\nRunning VQE optimization...")

result_vqe = minimize(
    cost_function,
    theta_init,
    method="COBYLA",
    options={
        "maxiter": 500,
        "rhobeg": 0.5
    }
)

print("\nOptimization finished")
print(f"Function evaluations: {len(energy_history)}")
print(f"VQE ground energy:    {result_vqe.fun:.6f}")
print(f"Exact ground energy:  {E0_exact:.6f}")
print(f"Absolute error:       {abs(result_vqe.fun - E0_exact):.6f}")
print(f"Relative error:       {abs(result_vqe.fun - E0_exact) / abs(E0_exact) * 100:.6f}%")
print("Optimal parameters:", result_vqe.x.round(4))


# -----------------------------
# 7. Plot convergence
# -----------------------------

plt.figure(figsize=(8, 5))
plt.plot(energy_history, label="VQE energy")
plt.axhline(E0_exact, linestyle="--", label=f"Exact E0 = {E0_exact:.4f}")
plt.xlabel("Optimizer iteration")
plt.ylabel("Energy")
plt.title("VQE Convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -----------------------------
# 8. Plot error
# -----------------------------

errors = [abs(e - E0_exact) for e in energy_history]

plt.figure(figsize=(8, 5))
plt.semilogy(errors)
plt.xlabel("Optimizer iteration")
plt.ylabel("|E(theta) - E0|")
plt.title("VQE Error Convergence")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()