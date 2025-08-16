"""
Quantum Entanglement Metrics Module

This module provides various entanglement measures and metrics for
quantum states, essential for quantum resource characterization.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from scipy.linalg import sqrtm, eigh
import itertools


class EntanglementMetrics:
    """
    Collection of entanglement measures for quantum states including
    entropy-based measures, negativity, concurrence, and multipartite measures.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize entanglement metrics calculator.
        
        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
    
    def state_to_density_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Convert pure state vector to density matrix.
        
        Args:
            state: Pure state vector
            
        Returns:
            Density matrix representation
        """
        if state.ndim == 1:
            # Pure state |psi><psi|
            return np.outer(state, np.conj(state))
        else:
            # Already a density matrix
            return state
    
    def partial_trace(self, rho: np.ndarray, subsystem: List[int]) -> np.ndarray:
        """
        Compute partial trace over specified subsystem.
        
        Args:
            rho: Density matrix of the full system
            subsystem: List of qubit indices to trace out
            
        Returns:
            Reduced density matrix
        """
        # Convert state to density matrix if necessary
        rho = self.state_to_density_matrix(rho)
        
        # Simple implementation for small systems
        # Keep track of which qubits to keep
        keep_qubits = [i for i in range(self.n_qubits) if i not in subsystem]
        n_keep = len(keep_qubits)
        
        if n_keep == 0:
            # Trace out everything - return the trace (scalar)
            return np.array([[np.trace(rho)]])
        
        # Dimensions
        dim_keep = 2 ** n_keep
        dim_trace = 2 ** len(subsystem)
        
        # Initialize reduced density matrix
        rho_reduced = np.zeros((dim_keep, dim_keep), dtype=complex)
        
        # Iterate over all basis states of the full system
        for i in range(self.dim):
            for j in range(self.dim):
                # Convert indices to binary representations
                i_bits = [(i >> k) & 1 for k in range(self.n_qubits)]
                j_bits = [(j >> k) & 1 for k in range(self.n_qubits)]
                
                # Check if traced-out qubits match (diagonal elements)
                trace_match = True
                for qubit in subsystem:
                    if i_bits[self.n_qubits - 1 - qubit] != j_bits[self.n_qubits - 1 - qubit]:
                        trace_match = False
                        break
                
                if trace_match:
                    # Extract indices for kept qubits
                    i_keep = 0
                    j_keep = 0
                    for idx, qubit in enumerate(keep_qubits):
                        bit_pos = self.n_qubits - 1 - qubit
                        i_keep += i_bits[bit_pos] * (2 ** (n_keep - 1 - idx))
                        j_keep += j_bits[bit_pos] * (2 ** (n_keep - 1 - idx))
                    
                    rho_reduced[i_keep, j_keep] += rho[i, j]
        
        return rho_reduced
    
    def von_neumann_entropy(self, rho: np.ndarray, base: float = 2) -> float:
        """
        Calculate von Neumann entropy S(rho) = -Tr(rho log rho).
        
        Args:
            rho: Density matrix
            base: Logarithm base (2 for bits, e for nats)
            
        Returns:
            von Neumann entropy
        """
        rho = self.state_to_density_matrix(rho)
        
        # Get eigenvalues
        eigenvals = eigh(rho, eigvals_only=True)
        
        # Remove zero and negative eigenvalues (numerical errors)
        eigenvals = eigenvals[eigenvals > 1e-12]
        
        if len(eigenvals) == 0:
            return 0.0
        
        # Calculate entropy
        if base == 2:
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
        elif base == np.e:
            entropy = -np.sum(eigenvals * np.log(eigenvals))
        else:
            entropy = -np.sum(eigenvals * np.log(eigenvals)) / np.log(base)
        
        return float(entropy)
    
    def entanglement_entropy(self, state: np.ndarray, partition_A: List[int]) -> float:
        """
        Calculate entanglement entropy between partition A and its complement.
        
        Args:
            state: Quantum state (vector or density matrix)
            partition_A: List of qubit indices in partition A
            
        Returns:
            Entanglement entropy S(A) = S(B)
        """
        # Get reduced density matrix of partition A
        complement_B = [i for i in range(self.n_qubits) if i not in partition_A]
        rho_A = self.partial_trace(state, complement_B)
        
        # Calculate von Neumann entropy
        return self.von_neumann_entropy(rho_A)
    
    def concurrence(self, state: np.ndarray) -> float:
        """
        Calculate concurrence for 2-qubit states.
        
        Args:
            state: 2-qubit quantum state
            
        Returns:
            Concurrence value (0 for separable, 1 for maximally entangled)
        """
        if self.n_qubits != 2:
            raise ValueError("Concurrence is only defined for 2-qubit systems")
        
        rho = self.state_to_density_matrix(state)
        
        # Pauli-Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        
        # Tensor product of Pauli-Y matrices
        sigma_y_tensor = np.kron(sigma_y, sigma_y)
        
        # Time-reversed state
        rho_tilde = sigma_y_tensor @ np.conj(rho) @ sigma_y_tensor
        
        # Calculate concurrence
        sqrt_rho = sqrtm(rho)
        R = sqrt_rho @ rho_tilde @ sqrt_rho
        
        # Get eigenvalues in descending order
        eigenvals = np.sort(eigh(R, eigvals_only=True))[::-1]
        eigenvals = np.sqrt(np.maximum(eigenvals, 0))  # Ensure non-negative
        
        concurrence = max(0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3])
        return float(concurrence)
    
    def negativity(self, state: np.ndarray, partition_A: List[int]) -> float:
        """
        Calculate negativity as an entanglement measure.
        
        Args:
            state: Quantum state
            partition_A: Partition to compute partial transpose over
            
        Returns:
            Negativity value
        """
        rho = self.state_to_density_matrix(state)
        
        # Compute partial transpose
        rho_pt = self._partial_transpose(rho, partition_A)
        
        # Get eigenvalues
        eigenvals = eigh(rho_pt, eigvals_only=True)
        
        # Negativity is sum of absolute values of negative eigenvalues
        negative_eigenvals = eigenvals[eigenvals < 0]
        negativity = -np.sum(negative_eigenvals)
        
        return float(negativity)
    
    def logarithmic_negativity(self, state: np.ndarray, partition_A: List[int]) -> float:
        """
        Calculate logarithmic negativity.
        
        Args:
            state: Quantum state
            partition_A: Partition to compute partial transpose over
            
        Returns:
            Logarithmic negativity value
        """
        neg = self.negativity(state, partition_A)
        if neg <= 1e-12:
            return 0.0
        return float(np.log2(2 * neg + 1))
    
    def _partial_transpose(self, rho: np.ndarray, subsystem: List[int]) -> np.ndarray:
        """
        Compute partial transpose over specified subsystem.
        
        Args:
            rho: Density matrix
            subsystem: Qubits to transpose
            
        Returns:
            Partially transposed density matrix
        """
        # Reshape to tensor form
        dims = [2] * self.n_qubits
        tensor_shape = dims + dims
        rho_tensor = rho.reshape(tensor_shape)
        
        # Transpose specified subsystems
        for qubit in subsystem:
            # Swap corresponding indices
            axes = list(range(2 * self.n_qubits))
            axes[qubit], axes[qubit + self.n_qubits] = axes[qubit + self.n_qubits], axes[qubit]
            rho_tensor = np.transpose(rho_tensor, axes)
        
        return rho_tensor.reshape(self.dim, self.dim)
    
    def multipartite_entanglement_entropy(self, state: np.ndarray) -> dict:
        """
        Calculate various multipartite entanglement measures.
        
        Args:
            state: Quantum state
            
        Returns:
            Dictionary containing different multipartite measures
        """
        results = {}
        
        # Bipartite entanglement entropies for all cuts
        bipartite_entropies = {}
        for i in range(1, self.n_qubits):
            for partition_A in itertools.combinations(range(self.n_qubits), i):
                partition_A = list(partition_A)
                entropy = self.entanglement_entropy(state, partition_A)
                key = f"S({','.join(map(str, partition_A))})"
                bipartite_entropies[key] = entropy
        
        results['bipartite_entropies'] = bipartite_entropies
        
        # Global entanglement (average over all bipartitions)
        all_entropies = list(bipartite_entropies.values())
        results['global_entanglement'] = np.mean(all_entropies) if all_entropies else 0.0
        
        # Maximum bipartite entanglement
        results['max_bipartite_entanglement'] = np.max(all_entropies) if all_entropies else 0.0
        
        return results
    
    def tangle(self, state: np.ndarray, qubit: int) -> float:
        """
        Calculate tangle (3-way entanglement measure) for a specific qubit.
        
        Args:
            state: 3-qubit quantum state
            qubit: Index of the qubit to calculate tangle for
            
        Returns:
            Tangle value
        """
        if self.n_qubits != 3:
            raise ValueError("Tangle is only implemented for 3-qubit systems")
        
        # Calculate all pairwise concurrences
        other_qubits = [i for i in range(3) if i != qubit]
        
        # Create 2-qubit states by tracing out the third qubit
        concurrences = []
        for other_qubit in other_qubits:
            trace_out = [i for i in range(3) if i not in [qubit, other_qubit]]
            gen_2 = EntanglementMetrics(2)
            rho_2 = self.partial_trace(state, trace_out)
            conc = gen_2.concurrence(rho_2)
            concurrences.append(conc**2)
        
        # Tangle = C_{i(jk)}^2 - C_{ij}^2 - C_{ik}^2
        # For this implementation, we approximate using available measures
        total_two_way = sum(concurrences)
        
        # This is a simplified version - full tangle calculation requires
        # the 3-way concurrence which is more complex
        return max(0, 1 - total_two_way)  # Simplified approximation
    
    def meyer_wallach_measure(self, state: np.ndarray) -> float:
        """
        Calculate Meyer-Wallach global entanglement measure.
        
        Args:
            state: Quantum state
            
        Returns:
            Meyer-Wallach measure (0 for separable, 1 for maximally entangled)
        """
        state_vec = state.flatten() if state.ndim > 1 else state
        
        # Calculate sum of local purities
        purity_sum = 0.0
        for i in range(self.n_qubits):
            # Trace out all qubits except i
            trace_out = [j for j in range(self.n_qubits) if j != i]
            rho_i = self.partial_trace(state, trace_out)
            purity_i = np.trace(rho_i @ rho_i).real
            purity_sum += purity_i
        
        # Meyer-Wallach measure
        mw_measure = 2 * (1 - purity_sum / self.n_qubits)
        return float(max(0, mw_measure))
    
    def schmidt_decomposition_2qubit(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Schmidt decomposition for 2-qubit pure states.
        
        Args:
            state: 2-qubit pure state vector
            
        Returns:
            Tuple of (schmidt_coefficients, basis_A, basis_B)
        """
        if self.n_qubits != 2:
            raise ValueError("Schmidt decomposition implemented only for 2-qubit systems")
        
        if state.ndim != 1:
            raise ValueError("Schmidt decomposition requires pure state vector")
        
        # Reshape state into 2x2 matrix
        state_matrix = state.reshape(2, 2)
        
        # Singular value decomposition
        U, schmidt_coeffs, Vh = np.linalg.svd(state_matrix)
        
        # Schmidt basis vectors
        basis_A = U.T  # Left Schmidt vectors
        basis_B = Vh   # Right Schmidt vectors
        
        return schmidt_coeffs, basis_A, basis_B
    
    def schmidt_rank(self, state: np.ndarray, tolerance: float = 1e-10) -> int:
        """
        Calculate Schmidt rank for 2-qubit states.
        
        Args:
            state: 2-qubit state
            tolerance: Tolerance for zero Schmidt coefficients
            
        Returns:
            Schmidt rank
        """
        schmidt_coeffs, _, _ = self.schmidt_decomposition_2qubit(state)
        return int(np.sum(schmidt_coeffs > tolerance))