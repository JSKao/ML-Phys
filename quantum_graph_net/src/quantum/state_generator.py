"""
Quantum State Generator Module

This module provides functionality to generate various quantum states
commonly used in quantum information and entanglement studies.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
import itertools


class QuantumStateGenerator:
    """
    Generator for various quantum states including GHZ, W, cluster states,
    and arbitrary graph states for quantum entanglement analysis.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize the quantum state generator.
        
        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        
    def computational_basis_state(self, bitstring: str) -> np.ndarray:
        """
        Generate a computational basis state |bitstring>.
        
        Args:
            bitstring: Binary string representation (e.g., "101")
            
        Returns:
            State vector as numpy array
        """
        if len(bitstring) != self.n_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != n_qubits {self.n_qubits}")
        
        state = np.zeros(self.dim, dtype=complex)
        index = int(bitstring, 2)  # Convert binary string to decimal
        state[index] = 1.0
        return state
    
    def plus_state(self) -> np.ndarray:
        """
        Generate |+>^n state (equal superposition).
        
        Returns:
            State vector as numpy array
        """
        # |+> = (|0> + |1>)/sqrt(2), so |+>^n = sum|x>/sqrt(2^n)
        state = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        return state
    
    def ghz_state(self) -> np.ndarray:
        """
        Generate GHZ state: (|00...0> + |11...1>)/sqrt(2).
        
        Returns:
            GHZ state vector
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1/np.sqrt(2)  # |00...0>
        state[-1] = 1/np.sqrt(2)  # |11...1>
        return state
    
    def w_state(self) -> np.ndarray:
        """
        Generate W state: symmetric superposition of all single-excitation states.
        (|100...0> + |010...0> + ... + |000...1>)/sqrt(n)
        
        Returns:
            W state vector
        """
        if self.n_qubits < 2:
            raise ValueError("W state requires at least 2 qubits")
        
        state = np.zeros(self.dim, dtype=complex)
        # Add all states with exactly one |1>
        for i in range(self.n_qubits):
            index = 1 << (self.n_qubits - 1 - i)  # Set i-th bit to 1
            state[index] = 1/np.sqrt(self.n_qubits)
        return state
    
    def bell_state(self, state_type: str = "phi_plus") -> np.ndarray:
        """
        Generate Bell states for 2-qubit systems.
        
        Args:
            state_type: One of "phi_plus", "phi_minus", "psi_plus", "psi_minus"
            
        Returns:
            Bell state vector
        """
        if self.n_qubits != 2:
            raise ValueError("Bell states are only defined for 2 qubits")
        
        state = np.zeros(4, dtype=complex)
        
        if state_type == "phi_plus":
            # |Phi+> = (|00> + |11>)/sqrt(2)
            state[0] = 1/np.sqrt(2)  # |00>
            state[3] = 1/np.sqrt(2)  # |11>
        elif state_type == "phi_minus":
            # |Phi-> = (|00> - |11>)/sqrt(2)
            state[0] = 1/np.sqrt(2)
            state[3] = -1/np.sqrt(2)
        elif state_type == "psi_plus":
            # |Psi+> = (|01> + |10>)/sqrt(2)
            state[1] = 1/np.sqrt(2)  # |01>
            state[2] = 1/np.sqrt(2)  # |10>
        elif state_type == "psi_minus":
            # |Psi-> = (|01> - |10>)/sqrt(2)
            state[1] = 1/np.sqrt(2)
            state[2] = -1/np.sqrt(2)
        else:
            raise ValueError(f"Unknown Bell state type: {state_type}")
        
        return state
    
    def cluster_state_1d(self) -> np.ndarray:
        """
        Generate 1D cluster state by applying CZ gates to |+>^n.
        
        Returns:
            1D cluster state vector
        """
        # Start with |+>^n
        state = self.plus_state()
        
        # Apply CZ gates between neighboring qubits
        for i in range(self.n_qubits - 1):
            state = self._apply_cz_gate(state, i, i + 1)
        
        return state
    
    def graph_state(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Generate arbitrary graph state from adjacency matrix.
        
        Args:
            adjacency_matrix: Symmetric binary matrix defining graph edges
            
        Returns:
            Graph state vector
        """
        if adjacency_matrix.shape != (self.n_qubits, self.n_qubits):
            raise ValueError("Adjacency matrix size doesn't match n_qubits")
        
        if not np.allclose(adjacency_matrix, adjacency_matrix.T):
            raise ValueError("Adjacency matrix must be symmetric")
        
        # Start with |+>^n
        state = self.plus_state()
        
        # Apply CZ gates for each edge in the graph
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if adjacency_matrix[i, j]:
                    state = self._apply_cz_gate(state, i, j)
        
        return state
    
    def random_pure_state(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random pure state using Haar measure.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Random pure state vector
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random complex amplitudes
        real_part = np.random.normal(0, 1, self.dim)
        imag_part = np.random.normal(0, 1, self.dim)
        state = real_part + 1j * imag_part
        
        # Normalize
        state = state / np.linalg.norm(state)
        return state
    
    def _apply_cz_gate(self, state: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """
        Apply controlled-Z gate between two qubits.
        
        Args:
            state: Current state vector
            qubit1: Control qubit index
            qubit2: Target qubit index
            
        Returns:
            State after CZ gate application
        """
        new_state = state.copy()
        
        # CZ gate flips phase when both qubits are |1>
        for i in range(self.dim):
            # Check if both qubits are |1> in computational basis
            bit1 = (i >> (self.n_qubits - 1 - qubit1)) & 1
            bit2 = (i >> (self.n_qubits - 1 - qubit2)) & 1
            
            if bit1 and bit2:
                new_state[i] *= -1
        
        return new_state
    
    def verify_normalization(self, state: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Verify that a quantum state is properly normalized.
        
        Args:
            state: Quantum state vector
            tolerance: Numerical tolerance for normalization check
            
        Returns:
            True if state is normalized
        """
        norm_squared = np.sum(np.abs(state)**2)
        return abs(norm_squared - 1.0) < tolerance
    
    def state_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate fidelity between two pure states.
        
        Args:
            state1, state2: Quantum state vectors
            
        Returns:
            Fidelity value between 0 and 1
        """
        overlap = np.abs(np.vdot(state1, state2))**2
        return overlap