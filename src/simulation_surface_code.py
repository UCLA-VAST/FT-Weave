import numpy as np
import stim
from typing import Tuple, Dict
from dataclasses import dataclass
from scipy.special import comb
import math
from scipy.optimize import minimize


@dataclass
class SimulationResult:
    """Container for simulation results"""

    infidelity: float
    physical_angle: float
    success_rate: float
    samples_per_weight: Dict[int, int]
    passes_per_weight: Dict[int, int]
    failure_counts: Dict[str, int]


class SurfaceCodeResourceState:
    """
    Implementation of Algorithm 1: Transversal multi-Pauli rotation protocol
    with optimal postselection for surface codes.
    """

    def __init__(
        self,
        code_distance: int,
        theta: float,
        physical_theta: float,
        p_ph: float,
        pauli_weight: int,
    ):
        """
        Initialize the surface code resource state preparation.

        Args:
            code_distance: Code distance d
            theta: Target rotation angle θ*
            physical_theta: physical rotation angle
            p_ph: Physical error rate
            pauli_weight: Weight m of multi-Pauli rotation (default: d)
        """
        self.d = code_distance
        self.theta = theta
        self.physical_theta = physical_theta
        self.rotation_weight = pauli_weight
        self.k = math.ceil(code_distance / pauli_weight)
        self.p_ph = p_ph

        tmp = self.compute_theta_n(0)
        assert np.isclose(self.theta, tmp), f"logical theta does not match input: {self.theta} <-> {tmp}"

        # Setup surface code layout
        self.setup_surface_code()

        # Define stabilizers for postselection regime
        self.setup_postselection_stabilizers()

    def setup_surface_code(self):
        """Setup the rotated surface code geometry and stabilizer structure."""
        # For a distance-d rotated surface code, we have d^2 data qubits
        # arranged in a diamond/checkerboard pattern
        self.n_data_qubits = self.d * self.d

        # X-stabilizers and Z-stabilizers
        self.x_stabilizers = []
        self.z_stabilizers = []
        self.x_stab_coords = []  # Track coordinates for filtering
        self.z_stab_coords = []

        # In rotated surface code:

        def qubit_index(row: int, col: int) -> int:
            """Convert (row, col) to qubit index."""
            idx = ((row - 1) // 2) * self.d + ((col - 1) // 2)
            assert idx >= 0, f"row: {row}, col: {col}, idx: {idx}"
            return idx

        bound = self.d * 2
        # Z stabilizers
        for row in range(0, bound + 1, 2):
            start_col = 2 + 2 * ((row // 2) % 2)
            for col in range(start_col, bound, 4):
                qubits: list[int | None] = [None, None, None, None]
                if row > 0:
                    # upper left neighbor
                    qubits[2] = qubit_index(row - 1, col - 1)
                    # upper right neighbor
                    qubits[0] = qubit_index(row - 1, col + 1)
                if row < bound:
                    # lower left neighbor
                    qubits[3] = qubit_index(row + 1, col - 1)
                    # lower right neighbor
                    qubits[1] = qubit_index(row + 1, col + 1)

                self.z_stabilizers.append(qubits)
                self.z_stab_coords.append((row, col))
        # X stabilizers
        for col in range(0, bound + 1, 2):
            start_row = 4 - 2 * ((col // 2) % 2)
            for row in range(start_row, bound, 4):
                qubits: list[int | None] = [None, None, None, None]
                if col > 0:
                    # upper left neighbor
                    qubits[1] = qubit_index(row - 1, col - 1)
                    # lower left neighbor
                    qubits[3] = qubit_index(row + 1, col - 1)
                if col < bound - 1:
                    # upper right neighbor
                    qubits[0] = qubit_index(row - 1, col + 1)
                    # lower right neighbor
                    qubits[2] = qubit_index(row + 1, col + 1)

                self.x_stabilizers.append(qubits)
                self.x_stab_coords.append((row, col))

        # Define Q_z: qubit set that forms support of Z_L
        self.Q_z = [row * self.d + row for row in range(self.d)]

    def setup_postselection_stabilizers(self):
        """Define S_PS: stabilizers in the postselection regime."""
        # Only check stabilizers in the first three rows for postselection
        self.S_PS_x = []
        self.S_PS_z = []

        # Filter X-stabilizers:
        for idx in range(len(self.x_stabilizers)):
            row, col = self.x_stab_coords[idx]
            if row == col or abs(row - col) == 4:
                self.S_PS_x.append(idx)
            elif abs(row - col) == 4:
                self.S_PS_x.append(idx)

        # Filter Z-stabilizers: only first 3 rows
        for idx in range(len(self.z_stabilizers)):
            row, col = self.z_stab_coords[idx]
            if row == col or abs(row - col) == 2:
                self.S_PS_z.append(idx + len(self.x_stabilizers))

        print(f"Rotated surface code d={self.d}:")
        print(f"  Total X-stabilizers: {len(self.x_stabilizers)}")
        print(f"  X-stabilizers Support: {self.x_stabilizers}")
        print(f"  X-stabilizers location: {self.x_stab_coords}")
        print(f"  Total Z-stabilizers: {len(self.z_stabilizers)}")
        print(f"  Z-stabilizers Support: {self.z_stabilizers}")
        print(f"  Z-stabilizers location: {self.z_stab_coords}")
        print(f"  S_PS X-stabilizers (first 3 rows): {self.S_PS_x}")
        print(f"  S_PS Z-stabilizers (first 3 rows): {self.S_PS_z}")
        print(f"  Q_z (logical Z support): {self.Q_z} qubits")

    def compute_u_coefficients(self, n: int) -> complex:
        """Compute u_n = i^n * sin^n(θ) * cos^(n)(k-θ) for weight-m rotation."""
        return (
            (1j) ** n
            * np.sin(self.physical_theta) ** n
            * np.cos(self.physical_theta) ** (self.k - n)
        )

    def compute_sampling_probability(self, n: int) -> float:
        """Compute q^sample_n = C(m,n) * (|u_n|^2 + |u_{m-n}|^2)."""

        if n > self.k:
            return 0.0
        u_n = self.compute_u_coefficients(n)
        u_kn = self.compute_u_coefficients(self.k - n)
        return comb(self.k, n, exact=True) * (np.abs(u_n) ** 2 + np.abs(u_kn) ** 2)

    def compute_theta_n(self, n: int) -> float:
        """Compute θ_n as per Eq. (C5) in the paper."""
        u_n = self.compute_u_coefficients(n)
        u_kn = self.compute_u_coefficients(self.k - n)

        denominator = np.sqrt(np.abs(u_n) ** 2 + np.abs(u_kn) ** 2)
        arg = np.abs(u_kn) / denominator
        return np.arcsin(arg)

    def create_initialization_circuit(self) -> stim.Circuit:
        """
        Algorithm 1, Line 1: Set all data physical qubits in |+⟩ state.
        """
        circuit = stim.Circuit()

        # Initialize in |+⟩
        circuit.append("RX", range(self.n_data_qubits))
        # Single-qubit depolarizing after initialization
        circuit.append("DEPOLARIZE1", range(self.n_data_qubits), self.p_ph)

        return circuit

    def measure_stabilizers(
        self,
        circuit: stim.Circuit,
    ) -> stim.Circuit:
        """
        Measure a set of stabilizers (X or Z type).

        Args:
            circuit: Circuit to append to

        Returns:
            Updated circuit
        """
        ancilla_offset = self.n_data_qubits

        x_stabilizer_indices = [
            ancilla_offset + i for i in range(len(self.x_stabilizers))
        ]
        ancilla_offset += len(self.x_stabilizers)
        z_stabilizer_indices = [
            ancilla_offset + i for i in range(len(self.z_stabilizers))
        ]

        # Initialize ancilla
        circuit.append("R", x_stabilizer_indices + z_stabilizer_indices)

        # For X stabilizer: H on ancilla
        circuit.append("H", x_stabilizer_indices)
        circuit.append("DEPOLARIZE1", x_stabilizer_indices, self.p_ph)

        for layer in range(4):  # 4 layers of CNOTs
            # X stabilizers
            for idx, support in enumerate(self.x_stabilizers):
                ancilla = x_stabilizer_indices[idx]
                if support[layer] is not None:
                    data_qubit = support[layer]
                    circuit.append("CX", [ancilla, data_qubit])
                    circuit.append("DEPOLARIZE2", [data_qubit, ancilla], self.p_ph)

            # Z stabilizers
            for idx, support in enumerate(self.z_stabilizers):
                ancilla = z_stabilizer_indices[idx]
                if support[layer] is not None:
                    data_qubit = support[layer]
                    circuit.append("CX", [data_qubit, ancilla])
                    circuit.append("DEPOLARIZE2", [ancilla, data_qubit], self.p_ph)
            circuit.append("TICK")
        # For X stabilizer: H on ancilla
        circuit.append("H", x_stabilizer_indices)
        circuit.append("DEPOLARIZE1", x_stabilizer_indices, self.p_ph)

        circuit.append("MR", x_stabilizer_indices + z_stabilizer_indices)

        # Measurements of interest for the detector are the x stabilizer

        return circuit

    def create_full_protocol_circuit(self, bit_string: np.ndarray) -> stim.Circuit:
        """
        Implement full Algorithm 1 protocol.

        Args:
            bit_string: Sampled bit string for syndrome subspace

        Returns:
            Complete circuit implementing the protocol
        """

        # Line 1: Initialize all qubits to |+⟩
        circuit: stim.Circuit = self.create_initialization_circuit()

        # Line 2: Measure stabilizer set S to generate |+⟩_L
        n_measurement = len(self.z_stabilizers) + len(self.x_stabilizers)

        circuit = self.measure_stabilizers(circuit)
        for i in range(
            1 + len(self.z_stabilizers),
            n_measurement + 1,
        ):
            circuit.append("DETECTOR", [stim.target_rec(-i)])

        # Line 6: Apply transversal multi-Pauli rotation on Q_z
        # We simulate this by applying Z^b based on sampled bit string
        z_qubit = [
            qubit
            for i, qubit in enumerate(self.Q_z[: len(bit_string)])
            if bit_string[i] == 1
        ]
        if z_qubit:
            circuit.append("Z_ERROR", z_qubit, 1)

        # Physical rotation gate (this introduces the non-Clifford component)
        # In practice, this would be R_z(θ) gate
        # We model errors on this operation
        circuit.append("DEPOLARIZE1", self.Q_z, self.p_ph)
        # Lines 7-11: Measure stabilizers twice for postselection
        for _ in range(2):
            circuit = self.measure_stabilizers(circuit)
            for i in self.S_PS_x:
                cur_index = i - n_measurement
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(cur_index),
                        stim.target_rec(cur_index - n_measurement),
                    ],
                )
            for i in self.S_PS_z:
                cur_index = i - n_measurement
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(cur_index),
                        stim.target_rec(cur_index - n_measurement),
                    ],
                )

        # print(repr(circuit))
        return circuit

    def check_postselection(self, measurements: np.ndarray) -> Tuple[bool, str]:
        """
        Check if measurement outcomes satisfy postselection criteria.

        Lines 3, 9-10: Check for unexpected syndromes in S_PS_x, S_PS_z.

        Args:
            measurements: Array of measurement outcomes

        Returns:
            (passed, failure_reason)
        """
        n_measurements = len(self.x_stabilizers) + len(self.z_stabilizers)
        if not np.all(measurements[: len(self.x_stabilizers)] == 0):
            return False, "init_syndrome"

        # Lines 9-10: Check two rounds of postselection measurements
        # Only check S_PS_x, S_PS_z stabilizers
        for round_num in range(2):
            start_idx = round_num * n_measurements + len(self.x_stabilizers)
            end_idx = start_idx + n_measurements
            round_measurements = measurements[start_idx:end_idx]
            # print(round_measurements)
            if not np.all(round_measurements == 0):
                return False, f"round_{round_num+1}_syndrome"

        return True, "success"

    def sample_bit_string(self) -> Tuple[np.ndarray, int]:
        """Sample bit string with correct probability distribution."""
        # Compute sampling probabilities for each Hamming weight
        probs = []
        for n in range(self.d + 1):
            probs.append(self.compute_sampling_probability(n))

        probs = np.array(probs)
        if np.sum(probs) < 1e-15:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / np.sum(probs)

        # Sample Hamming weight
        hamming_weight = np.random.choice(self.d + 1, p=probs)
        # Generate random bit string with sampled Hamming weight
        bit_string = np.zeros(self.d, dtype=int)
        if hamming_weight > 0:
            positions = np.random.choice(self.d, size=hamming_weight, replace=False)
            bit_string[positions] = 1

        return bit_string, hamming_weight

    def run_single_trial(self) -> Tuple[bool, int, str]:
        """
        Run single trial of Algorithm 1.

        Returns:
            (passed, hamming_weight, failure_reason)
        """
        # Sample bit string for syndrome subspace
        bit_string, n = self.sample_bit_string()
        # print(f"n: {n}, bit string: {bit_string}")
        # Create and simulate circuit
        circuit = self.create_full_protocol_circuit(bit_string)
        # print(repr(circuit))
        # Run simulation
        sampler = circuit.compile_detector_sampler()
        measurements = sampler.sample(shots=1)[0]
        # print(measurements)
        # Check postselection
        passed, failure_reason = self.check_postselection(measurements)

        return passed, n, failure_reason

    def run_simulation(self, n_shots: int = 10000) -> SimulationResult:
        """
        Run full simulation with statistics collection.

        Args:
            n_shots: Number of Monte Carlo trials

        Returns:
            SimulationResult with infidelity and success rate
        """
        # Initialize statistics
        samples_per_weight = {n: 0 for n in range(self.k + 1)}
        passes_per_weight = {n: 0 for n in range(self.k + 1)}
        failure_counts = {
            "init_syndrome": 0,
            "round_1_syndrome": 0,
            "round_2_syndrome": 0,
            "success": 0,
        }

        print(
            f"Running {n_shots} shots for d={self.d}, θ={self.theta:.4f}, p_ph={self.p_ph}..."
        )

        for shot in range(n_shots):
            if (shot + 1) % 1000 == 0:
                print(f"  Progress: {shot+1}/{n_shots}")

            passed, n, failure_reason = self.run_single_trial()

            samples_per_weight[n] += 1
            failure_counts[failure_reason] += 1

            if passed:
                passes_per_weight[n] += 1

        # Compute final metrics following Eq. (C5) methodology

        # print("samples_per_weight")
        # print(samples_per_weight)
        # print("passes_per_weight")
        # print(passes_per_weight)

        total_infidelity = 0.0
        p_suc = 0
        for n in range(self.k):
            N_n_sample = samples_per_weight[n]
            N_n_pass = passes_per_weight[n]

            if N_n_sample > 0 and N_n_pass > 0:
                q_sample_n = self.compute_sampling_probability(n)
                q_pass_n = N_n_pass / N_n_sample

                # Compute infidelity contribution
                theta_n = self.compute_theta_n(n)
                F_n = np.sin(theta_n - self.theta) ** 2
                # print(n)
                # print(
                #     np.sin(self.physical_theta) ** (2 * self.k)
                #     + np.cos(self.physical_theta) ** (2 * self.k)
                # )
                # print(q_sample_n)
                # print(q_pass_n)
                # print(F_n)
                # print(q_sample_n * q_pass_n * F_n)
                # input()
                total_infidelity += q_sample_n * q_pass_n * F_n

                # Success rate
                p_suc += q_sample_n * q_pass_n
        # Normalize infidelity
        if p_suc > 0:
            infidelity = total_infidelity / p_suc
        else:
            infidelity = 1.0

        print(f"\n{'='*60}")
        print("Simulation Results:")
        print(f"{'='*60}")
        print(f"Total infidelity: {total_infidelity:.6E}")
        print(f"Success rate (p_suc): {p_suc:.6E}")
        print(f"Infidelity (1-F): {infidelity:.8E}")
        print(f"Fidelity (F): {1-infidelity:.8E}")
        print("\nFailure breakdown:")
        for reason, count in failure_counts.items():
            print(f"  {reason}: {count} ({count/n_shots*100:.2f}%)")

        return SimulationResult(
            infidelity=infidelity,
            physical_angle=self.physical_theta,
            success_rate=p_suc,
            samples_per_weight=samples_per_weight,
            passes_per_weight=passes_per_weight,
            failure_counts=failure_counts,
        )


if __name__ == "__main__":
    # Parameters for T gate preparation (π/8 rotation)
    code_distance = 3  # Use d=5 for rotated surface code
    m = 1
    physical_thetas = [0.1]
    physical_thetas = [0.221351] 
    # physical_thetas = [0.1, 0.001, 0.001]
    for physical_theta in physical_thetas:
        k = math.ceil(code_distance / m)
        p_ideal = np.sin(physical_theta) ** (2 * k) + np.cos(physical_theta) ** (2 * k)
        theta = np.arcsin(np.sin(physical_theta) ** (k) / np.sqrt(p_ideal))
        # p_ph = 0.001  # Physical error rate
        p_ph = 0.003  # Physical error rate

        print("=" * 60)
        print("Rotated Surface Code Resource State Preparation")
        print(f"physical rotation:{physical_theta}, logical rotation: {theta}")
        print("=" * 60)

        # Create simulator
        sim = SurfaceCodeResourceState(
            code_distance=code_distance,
            theta=theta,
            physical_theta=physical_theta,
            p_ph=p_ph,
            pauli_weight=m,
        )

        # Run simulation
        result = sim.run_simulation(n_shots=100)

        # Detailed analysis
        print(f"\n{'='*60}")
        print("Detailed Statistics:")
        print("=" * 60)
        print("Samples per Hamming weight:")
        for n, count in result.samples_per_weight.items():
            if count > 0:
                prob = sim.compute_sampling_probability(n)
                print(f"  n={n}: {count} samples (expected prob: {prob:.4f})")

        print("\nPasses per Hamming weight:")
        for n, count in result.passes_per_weight.items():
            if count > 0 and result.samples_per_weight[n] > 0:
                pass_rate = count / result.samples_per_weight[n]
                print(f"  n={n}: {count} passes ({pass_rate*100:.2f}%)")

        # Compute theoretical leading order
        print("=" * 60)
        print("Leading Order Analysis (n=1):")
        print("=" * 60)
        if result.samples_per_weight[1] > 0:
            q_sample_1 = sim.compute_sampling_probability(1)
            q_pass_1 = result.passes_per_weight[1] / result.samples_per_weight[1]
            theta_1 = sim.compute_theta_n(1)

            print(f"q_sample_1 = {q_sample_1:.6E}")
            print(f"q_pass_1 = {q_pass_1:.6E}")
            print(f"θ_1 = {theta_1:.6} rad")
            print(f"sin²(θ_1 - θ*) = {np.sin(theta_1 - theta)**2:.8E}\n\n")
