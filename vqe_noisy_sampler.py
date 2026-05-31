import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import transpile
from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import SamplerV2


hamiltonian = SparsePauliOp.from_list([
    ("ZZ", -1.052373245772859),
    ("ZI", 0.39793742484318045),
    ("IZ", -0.39793742484318045),
    ("XX", -0.01128010425623538)
])

E0_exact = -1.137270174


# Updated ansatz using n_local instead of TwoLocal
ansatz = n_local(
    num_qubits=2,
    rotation_blocks=["ry", "rz"],
    entanglement_blocks="cx",
    reps=2,
    entanglement="full"
)

ansatz = transpile(
    ansatz,
    basis_gates=["rx", "ry", "rz", "cx"]
)


def build_noise_model(p):
    noise_model = NoiseModel()

    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(p, 1),
        ["ry", "rz"]
    )

    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(10 * p, 2),
        ["cx"]
    )

    readout_error = ReadoutError([
        [1 - p / 2, p / 2],
        [p / 2, 1 - p / 2]
    ])

    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def measure_pauli(circuit, pauli):
    qc = circuit.copy()

    for i, p in enumerate(pauli):
        if p == "X":
            qc.h(i)
        elif p == "Y":
            qc.sdg(i)
            qc.h(i)

    qc.measure_all()
    return qc


def expectation_from_counts(counts, pauli):
    exp = 0
    shots = sum(counts.values())

    for bitstring, count in counts.items():
        parity = 1
        bitstring = bitstring[::-1]

        for i, p in enumerate(pauli):
            if p != "I":
                if bitstring[i] == "1":
                    parity *= -1

        exp += parity * count

    return exp / shots


def run_vqe_sampler(noise_level, n_iter=200):
    noise_model = build_noise_model(noise_level)

    sampler = SamplerV2(
        options={
            "backend_options": {
                "noise_model": noise_model
            },
            "run_options": {
                "shots": 4096
            }
        }
    )

    history = []

    def cost_fn(params):
        bound = ansatz.assign_parameters(params)

        energy = 0.0

        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            pauli_label = pauli.to_label()

            qc = measure_pauli(bound, pauli_label)
            qc = transpile(
                qc,
                basis_gates=["rx", "ry", "rz", "cx"]
            )

            result = sampler.run([qc]).result()
            counts = result[0].data.meas.get_counts()

            exp_val = expectation_from_counts(counts, pauli_label)
            energy += coeff.real * exp_val

        history.append(float(energy))
        return float(energy)

    np.random.seed(7)
    theta0 = np.random.uniform(
        -np.pi,
        np.pi,
        ansatz.num_parameters
    )

    result = minimize(
        cost_fn,
        theta0,
        method="COBYLA",
        options={
            "maxiter": n_iter
        }
    )

    return result.fun, history


noise_levels = [0.0, 0.02, 0.05]

plt.figure(figsize=(10, 5))

for noise_level in noise_levels:
    final_energy, history = run_vqe_sampler(
        noise_level,
        n_iter=200
    )

    label = "noiseless" if noise_level == 0 else f"noise = {noise_level:.0%}"

    plt.plot(history, label=label)

    print(f"{label}: final energy = {final_energy:.6f}")


plt.axhline(
    E0_exact,
    linestyle="--",
    color="black",
    label="Exact"
)

plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("SamplerV2-based Noisy VQE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()