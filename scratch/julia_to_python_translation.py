import numpy as np
from typing import List, Tuple, Dict
import stim

class RotatedSurfaceCodeTMR:
    """
    Translation of Julia code for TMR on rotated surface code.
    Key difference: Uses per-qubit Bernoulli sampling instead of syndrome subspace sampling.
    """
    
    def __init__(self, distance: int, theta: float, p_t: float):
        """
        Args:
            distance: Code distance
            theta: Rotation angle
            p_t: Physical error rate (uniform_data and uniform_anc)
        """
        self.dist = distance
        self.theta = theta
        self.p_t = p_t
        
        # Setup surface code
        self.setup_rotated_surface_code()
        
        # Logical Z qubits (diagonal)
        self.logical_z_qubits = [
            (self.dist - lz, lz - 1) for lz in range(1, self.dist + 1)
        ]
        
        # Calculate ideal angle
        self.calculate_ideal_angle()
        
    def setup_rotated_surface_code(self):
        """Setup rotated surface code geometry."""
        # Data qubits on a grid
        self.data_qubits = {}
        idx = 0
        for x in range(self.dist):
            for y in range(self.dist):
                self.data_qubits[(x, y)] = idx
                idx += 1
        
        # Z-stabilizers (even parity: x+y even)
        self.z_plaqs = [
            (x, y) 
            for x in range(self.dist - 1)
            for y in range(self.dist + 1)
            if (x + y) % 2 == 0
        ]
        
        # X-stabilizers (odd parity: x+y odd)
        self.x_plaqs = [
            (x, y)
            for x in range(self.dist + 1)
            for y in range(self.dist - 1)
            if (x + y) % 2 == 1
        ]
        
        # Create plaquette info
        self.z_plaq_info = self._make_plaq_info(self.z_plaqs)
        self.x_plaq_info = self._make_plaq_info(self.x_plaqs)
        
        # Diagonal X ancillas (touching logical Z qubits)
        self.diagonal_x_ancilla = self._find_diagonal_x_plaqs()
        
    def _make_plaq_info(self, plaqs: List[Tuple[int, int]]) -> List[Dict]:
        """Create plaquette information."""
        plaq_info = []
        for (x, y) in plaqs:
            qubits = []
            for (xx, yy) in [(x, y), (x, y+1), (x+1, y), (x+1, y+1)]:
                if (xx, yy) in self.data_qubits:
                    qubits.append(self.data_qubits[(xx, yy)])
            
            if len(qubits) >= 2:
                plaq_info.append({
                    'coords': (x, y),
                    'data': qubits
                })
        return plaq_info
    
    def _find_diagonal_x_plaqs(self) -> List[int]:
        """Find X-stabilizers that touch logical Z qubits."""
        lz_qubits = set(self.data_qubits[coord] for coord in self.logical_z_qubits)
        diagonal_plaqs = []
        
        for idx, plaq in enumerate(self.x_plaq_info):
            if len(plaq['data']) == 4:
                for q in plaq['data']:
                    if q in lz_qubits:
                        diagonal_plaqs.append(idx)
                        break
        return diagonal_plaqs
    
    def calculate_ideal_angle(self):
        """Calculate ideal rotation angle (theta_L)."""
        amp_I = np.cos(self.theta / 2)
        amp_Z = np.sin(self.theta / 2)
        
        # Ideal: all Z operators applied (weight = dist)
        ideal_op = (amp_Z ** self.dist) / np.sqrt(
            amp_I ** (2 * self.dist) + amp_Z ** (2 * self.dist)
        )
        self.ideal_i = (1j) ** self.dist
        self.ideal_angle = 2 * np.arcsin(ideal_op)
        
    def rz_gate_sampling(self, state: stim.TableauSimulator) -> Tuple[np.ndarray, float, float]:
        """
        KEY DIFFERENCE FROM PREVIOUS IMPLEMENTATION:
        
        This samples EACH QUBIT INDEPENDENTLY with Bernoulli distribution.
        p(Z on qubit i) = sin²(θ/2)
        p(I on qubit i) = cos²(θ/2)
        
        This gives different statistics than syndrome subspace sampling!
        """
        pI = np.cos(self.theta / 2) ** 2
        pZ = np.sin(self.theta / 2) ** 2
        
        projected = np.zeros(self.dist, dtype=int)
        
        # Sample each qubit independently
        for i in range(self.dist):
            r = np.random.random()
            if r >= pI:  # Apply Z with probability pZ
                projected[i] = 1
                coord = self.logical_z_qubits[i]
                qubit_idx = self.data_qubits[coord]
                # Apply Z gate in Stim
                state.z(qubit_idx)
        
        return projected, pI, pZ
    
    def compute_actual_angle(self, projected: np.ndarray) -> Tuple[float, complex]:
        """
        Compute actual rotation angle based on sampled bit string.
        
        This matches the Julia code calculation.
        """
        amp_I = np.cos(self.theta / 2)
        amp_Z = np.sin(self.theta / 2)
        
        weight = np.sum(projected)
        
        # Take min/max to count I^d and Z^d correctly
        a = min(weight, self.dist - weight)
        b = max(weight, self.dist - weight)
        
        # Actual logical Z expected value
        actual_op = (amp_I ** a * amp_Z ** b) / np.sqrt(
            amp_I ** (2*b) * amp_Z ** (2*a) + amp_Z ** (2*b) * amp_I ** (2*a)
        )
        
        # Actual angle
        actual_angle = 2 * np.arcsin(actual_op)
        
        # Phase factor
        actual_i = (1j) ** (b - a)
        
        return actual_angle, actual_i
    
    def compare_sampling_methods(self, n_samples: int = 10000):
        """
        Compare the two sampling methods:
        1. Per-qubit Bernoulli (Julia code)
        2. Syndrome subspace (previous Python code)
        """
        print("="*70)
        print("Comparing Sampling Methods")
        print("="*70)
        
        # Method 1: Per-qubit Bernoulli (Julia style)
        pI = np.cos(self.theta / 2) ** 2
        pZ = np.sin(self.theta / 2) ** 2
        
        weight_counts_bernoulli = {n: 0 for n in range(self.dist + 1)}
        
        for _ in range(n_samples):
            projected = np.zeros(self.dist, dtype=int)
            for i in range(self.dist):
                if np.random.random() >= pI:
                    projected[i] = 1
            weight = int(np.sum(projected))
            weight_counts_bernoulli[weight] += 1
        
        # Method 2: Syndrome subspace sampling (previous code)
        from scipy.special import comb
        
        def compute_u(n):
            return (1j)**n * np.sin(self.theta)**n * np.cos(self.theta)**(self.dist - n)
        
        def q_sample(n):
            u_n = compute_u(n)
            u_dn = compute_u(self.dist - n)
            return comb(self.dist, n, exact=True) * (np.abs(u_n)**2 + np.abs(u_dn)**2)
        
        # Compute probabilities
        probs_subspace = [q_sample(n) for n in range(self.dist + 1)]
        probs_subspace = np.array(probs_subspace) / np.sum(probs_subspace)
        
        weight_counts_subspace = {n: 0 for n in range(self.dist + 1)}
        for _ in range(n_samples):
            n = np.random.choice(self.dist + 1, p=probs_subspace)
            weight_counts_subspace[n] += 1
        
        # Method 3: Theoretical Binomial
        weight_counts_binomial = {}
        for n in range(self.dist + 1):
            prob = comb(self.dist, n, exact=True) * (pZ ** n) * (pI ** (self.dist - n))
            weight_counts_binomial[n] = prob * n_samples
        
        # Display results
        print(f"\nRotation angle θ = {self.theta:.4f} rad = {np.degrees(self.theta):.2f}°")
        print(f"Distance d = {self.dist}")
        print(f"pI = cos²(θ/2) = {pI:.6f}")
        print(f"pZ = sin²(θ/2) = {pZ:.6f}")
        print(f"\nSamples: {n_samples}")
        print("\n" + "="*70)
        print(f"{'Weight':<8} {'Bernoulli':<15} {'Subspace':<15} {'Binomial':<15}")
        print(f"{'n':<8} {'(Julia)':<15} {'(Python)':<15} {'(Theory)':<15}")
        print("="*70)
        
        for n in range(self.dist + 1):
            bernoulli_pct = weight_counts_bernoulli[n] / n_samples * 100
            subspace_pct = weight_counts_subspace[n] / n_samples * 100
            binomial_pct = weight_counts_binomial[n] / n_samples * 100
            
            print(f"{n:<8} {bernoulli_pct:<14.2f}% {subspace_pct:<14.2f}% {binomial_pct:<14.2f}%")
        
        print("\n" + "="*70)
        print("KEY INSIGHT:")
        print("="*70)
        print("• Bernoulli (Julia): Each qubit sampled independently")
        print(f"  → Binomial(d={self.dist}, p={pZ:.4f}) distribution")
        print(f"  → Higher weights MORE likely for θ > π/4")
        print()
        print("• Subspace (Previous): Samples syndrome subspaces")
        print(f"  → Includes quantum coherence between |b⟩ and |b̄⟩")
        print(f"  → Symmetric around d/2")
        print()
        print("For θ = π/8 (T gate): pZ = 0.0732 << 0.5")
        print("  → Lower weights strongly preferred in BOTH methods")
        print("  → Higher orders (n≥2) have lower sampling probability")
        print("="*70)


# Example usage
if __name__ == "__main__":
    # T gate parameters
    distance = 5
    theta = np.pi / 8  # T gate
    p_t = 0.001
    
    sim = RotatedSurfaceCodeTMR(distance, theta, p_t)
    
    # Compare sampling methods
    sim.compare_sampling_methods(n_samples=10000)
    
    print("\n" + "="*70)
    print("ANSWER TO YOUR QUESTION:")
    print("="*70)
    print()
    print("Why can we ignore higher-order terms?")
    print()
    print("The Julia code uses PER-QUBIT Bernoulli sampling:")
    print(f"  • p(Z on each qubit) = sin²(θ/2) = {np.sin(theta/2)**2:.6f}")
    print(f"  • For d={distance}, weight follows Binomial({distance}, {np.sin(theta/2)**2:.4f})")
    print()
    print("For small θ (like π/8 for T gate):")
    print(f"  • P(weight=0) ≈ {np.cos(theta/2)**(2*distance):.4f}")
    print(f"  • P(weight=1) ≈ {distance * np.sin(theta/2)**2 * np.cos(theta/2)**(2*(distance-1)):.4f}")
    print(f"  • P(weight=2) ≈ {distance*(distance-1)/2 * np.sin(theta/2)**4 * np.cos(theta/2)**(2*(distance-2)):.6f}")
    print()
    print("Higher weights ARE SUPPRESSED in sampling probability!")
    print()
    print("Combined with postselection suppression q_pass_n ~ O(p_ph^{n-1}):")
    print("  • Total contribution: q_n = q_sample_n × q_pass_n")
    print("  • For n≥2: Both factors are small → negligible contribution")
    print("="*70)
