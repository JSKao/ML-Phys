import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_graph_net.src.quantum.state_generator import QuantumStateGenerator


class TestQuantumStateGenerator:
    """Test suite for QuantumStateGenerator functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.gen_2 = QuantumStateGenerator(2)
        self.gen_3 = QuantumStateGenerator(3)
        self.gen_4 = QuantumStateGenerator(4)
    
    @pytest.mark.quantum
    def test_initialization(self):
        """Test proper initialization of generator"""
        assert self.gen_2.n_qubits == 2
        assert self.gen_2.dim == 4
        assert self.gen_3.n_qubits == 3
        assert self.gen_3.dim == 8
    
    @pytest.mark.quantum
    def test_computational_basis_state(self):
        """Test computational basis state generation"""
        # Test |00⟩
        state_00 = self.gen_2.computational_basis_state("00")
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(state_00, expected)
        
        # Test |11⟩
        state_11 = self.gen_2.computational_basis_state("11")
        expected = np.array([0, 0, 0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(state_11, expected)
        
        # Test normalization
        assert self.gen_2.verify_normalization(state_00)
        assert self.gen_2.verify_normalization(state_11)
    
    @pytest.mark.quantum
    def test_computational_basis_wrong_length(self):
        """Test error handling for wrong bitstring length"""
        with pytest.raises(ValueError, match="Bitstring length"):
            self.gen_2.computational_basis_state("101")
    
    @pytest.mark.quantum
    def test_plus_state(self):
        """Test |+⟩^⊗n state generation"""
        plus_2 = self.gen_2.plus_state()
        expected = np.ones(4) / 2  # (1,1,1,1)/2
        np.testing.assert_array_almost_equal(plus_2, expected)
        
        # Test normalization
        assert self.gen_2.verify_normalization(plus_2)
        
        # Test 3-qubit case
        plus_3 = self.gen_3.plus_state()
        expected_3 = np.ones(8) / np.sqrt(8)
        np.testing.assert_array_almost_equal(plus_3, expected_3)
        assert self.gen_3.verify_normalization(plus_3)
    
    @pytest.mark.quantum
    def test_ghz_state(self):
        """Test GHZ state generation"""
        ghz_2 = self.gen_2.ghz_state()
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(ghz_2, expected)
        
        # Test normalization
        assert self.gen_2.verify_normalization(ghz_2)
        
        # Test 3-qubit GHZ
        ghz_3 = self.gen_3.ghz_state()
        expected_3 = np.zeros(8, dtype=complex)
        expected_3[0] = 1/np.sqrt(2)  # |000⟩
        expected_3[7] = 1/np.sqrt(2)  # |111⟩
        np.testing.assert_array_almost_equal(ghz_3, expected_3)
        assert self.gen_3.verify_normalization(ghz_3)
    
    @pytest.mark.quantum
    def test_w_state(self):
        """Test W state generation"""
        # W state not defined for 1 qubit
        gen_1 = QuantumStateGenerator(1)
        with pytest.raises(ValueError, match="W state requires at least 2 qubits"):
            gen_1.w_state()
        
        # Test 2-qubit W state
        w_2 = self.gen_2.w_state()
        expected = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
        np.testing.assert_array_almost_equal(w_2, expected)
        assert self.gen_2.verify_normalization(w_2)
        
        # Test 3-qubit W state
        w_3 = self.gen_3.w_state()
        expected_3 = np.zeros(8, dtype=complex)
        expected_3[1] = 1/np.sqrt(3)  # |001⟩
        expected_3[2] = 1/np.sqrt(3)  # |010⟩
        expected_3[4] = 1/np.sqrt(3)  # |100⟩
        np.testing.assert_array_almost_equal(w_3, expected_3)
        assert self.gen_3.verify_normalization(w_3)
    
    @pytest.mark.quantum
    def test_bell_states(self):
        """Test all Bell states"""
        # Test wrong number of qubits
        with pytest.raises(ValueError, match="Bell states are only defined for 2 qubits"):
            self.gen_3.bell_state()
        
        # Test |Φ+⟩
        phi_plus = self.gen_2.bell_state("phi_plus")
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(phi_plus, expected)
        
        # Test |Φ-⟩
        phi_minus = self.gen_2.bell_state("phi_minus")
        expected = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex)
        np.testing.assert_array_almost_equal(phi_minus, expected)
        
        # Test |Ψ+⟩
        psi_plus = self.gen_2.bell_state("psi_plus")
        expected = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
        np.testing.assert_array_almost_equal(psi_plus, expected)
        
        # Test |Ψ-⟩
        psi_minus = self.gen_2.bell_state("psi_minus")
        expected = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
        np.testing.assert_array_almost_equal(psi_minus, expected)
        
        # Test all are normalized
        for state in [phi_plus, phi_minus, psi_plus, psi_minus]:
            assert self.gen_2.verify_normalization(state)
        
        # Test unknown state type
        with pytest.raises(ValueError, match="Unknown Bell state type"):
            self.gen_2.bell_state("unknown")
    
    @pytest.mark.quantum
    def test_cluster_state_1d(self):
        """Test 1D cluster state generation"""
        cluster_2 = self.gen_2.cluster_state_1d()
        
        # 2-qubit cluster state = 1/2 * (|00⟩ + |01⟩ + |10⟩ - |11⟩)
        expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
        np.testing.assert_array_almost_equal(cluster_2, expected)
        assert self.gen_2.verify_normalization(cluster_2)
        
        # Test 3-qubit cluster state
        cluster_3 = self.gen_3.cluster_state_1d()
        assert self.gen_3.verify_normalization(cluster_3)
    
    @pytest.mark.quantum
    def test_graph_state(self):
        """Test arbitrary graph state generation"""
        # Test with 2-qubit complete graph (same as cluster state)
        adj_complete = np.array([[0, 1], [1, 0]])
        graph_state = self.gen_2.graph_state(adj_complete)
        cluster_state = self.gen_2.cluster_state_1d()
        np.testing.assert_array_almost_equal(graph_state, cluster_state)
        
        # Test with no edges (should be |+⟩^⊗n)
        adj_empty = np.zeros((3, 3))
        graph_state_empty = self.gen_3.graph_state(adj_empty)
        plus_state = self.gen_3.plus_state()
        np.testing.assert_array_almost_equal(graph_state_empty, plus_state)
        
        # Test wrong adjacency matrix size
        with pytest.raises(ValueError, match="Adjacency matrix size"):
            self.gen_2.graph_state(np.zeros((3, 3)))
        
        # Test non-symmetric adjacency matrix
        adj_nonsym = np.array([[0, 1], [0, 0]])
        with pytest.raises(ValueError, match="Adjacency matrix must be symmetric"):
            self.gen_2.graph_state(adj_nonsym)
    
    @pytest.mark.quantum
    def test_random_pure_state(self):
        """Test random pure state generation"""
        # Test reproducibility with seed
        random_1 = self.gen_2.random_pure_state(seed=42)
        random_2 = self.gen_2.random_pure_state(seed=42)
        np.testing.assert_array_almost_equal(random_1, random_2)
        
        # Test normalization
        assert self.gen_2.verify_normalization(random_1)
        
        # Test different seeds give different states
        random_3 = self.gen_2.random_pure_state(seed=123)
        assert not np.allclose(random_1, random_3)
    
    @pytest.mark.quantum
    def test_verify_normalization(self):
        """Test normalization verification"""
        # Test normalized state
        normalized = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        assert self.gen_2.verify_normalization(normalized)
        
        # Test non-normalized state
        non_normalized = np.array([1, 1, 0, 0])
        assert not self.gen_2.verify_normalization(non_normalized)
        
        # Test with tolerance
        almost_normalized = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 1e-12])
        assert self.gen_2.verify_normalization(almost_normalized, tolerance=1e-10)
        assert not self.gen_2.verify_normalization(almost_normalized, tolerance=1e-16)
    
    @pytest.mark.quantum
    def test_state_fidelity(self):
        """Test fidelity calculation between states"""
        # Test fidelity with itself
        state = self.gen_2.ghz_state()
        assert abs(self.gen_2.state_fidelity(state, state) - 1.0) < 1e-10
        
        # Test fidelity between orthogonal states
        state_00 = self.gen_2.computational_basis_state("00")
        state_11 = self.gen_2.computational_basis_state("11")
        assert abs(self.gen_2.state_fidelity(state_00, state_11)) < 1e-10
        
        # Test fidelity between Bell states
        phi_plus = self.gen_2.bell_state("phi_plus")
        phi_minus = self.gen_2.bell_state("phi_minus")
        fidelity = self.gen_2.state_fidelity(phi_plus, phi_minus)
        assert abs(fidelity - 0.0) < 1e-10  # Should be 0 for orthogonal states