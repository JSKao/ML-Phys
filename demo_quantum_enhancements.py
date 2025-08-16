#!/usr/bin/env python3
"""
Demonstration of Enhanced Quantum ML-Phys Capabilities

This script demonstrates the newly implemented quantum state generation
and entanglement analysis capabilities.
"""

import numpy as np
import sys
import os
sys.path.append('.')

from quantum_graph_net.src.quantum.state_generator import QuantumStateGenerator
from quantum_graph_net.src.quantum.entanglement_metrics import EntanglementMetrics


def demonstrate_quantum_states():
    """Demonstrate quantum state generation capabilities"""
    print("=" * 60)
    print("🚀 QUANTUM STATE GENERATION DEMO")
    print("=" * 60)
    
    # Initialize generators
    gen_2 = QuantumStateGenerator(2)
    gen_3 = QuantumStateGenerator(3)
    
    print("\n1. Bell State Generation:")
    for bell_type in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
        bell_state = gen_2.bell_state(bell_type)
        print(f"   |{bell_type}⟩: {bell_state}")
    
    print("\n2. GHZ States:")
    ghz_2 = gen_2.ghz_state()
    ghz_3 = gen_3.ghz_state()
    print(f"   2-qubit GHZ: {ghz_2}")
    print(f"   3-qubit GHZ: {ghz_3}")
    
    print("\n3. W States:")
    w_2 = gen_2.w_state()
    w_3 = gen_3.w_state()
    print(f"   2-qubit W: {w_2}")
    print(f"   3-qubit W: {w_3}")
    
    print("\n4. Graph States:")
    # Linear cluster state
    cluster_3 = gen_3.cluster_state_1d()
    print(f"   3-qubit cluster: {cluster_3}")
    
    # Custom graph state
    adj_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle
    triangle_state = gen_3.graph_state(adj_matrix)
    print(f"   Triangle graph: {triangle_state}")


def demonstrate_entanglement_analysis():
    """Demonstrate entanglement analysis capabilities"""
    print("\n" + "=" * 60)
    print("🔬 ENTANGLEMENT ANALYSIS DEMO")
    print("=" * 60)
    
    # Initialize analyzers
    gen_2 = QuantumStateGenerator(2)
    gen_3 = QuantumStateGenerator(3)
    metrics_2 = EntanglementMetrics(2)
    metrics_3 = EntanglementMetrics(3)
    
    print("\n1. Entanglement Entropy Analysis:")
    
    # Bell state
    bell_state = gen_2.bell_state("phi_plus")
    entropy_bell = metrics_2.entanglement_entropy(bell_state, [0])
    print(f"   Bell state entropy: {entropy_bell:.6f} bits")
    
    # Product state
    product_state = gen_2.computational_basis_state("00")
    entropy_product = metrics_2.entanglement_entropy(product_state, [0])
    print(f"   Product state entropy: {entropy_product:.6f} bits")
    
    # GHZ state
    ghz_3 = gen_3.ghz_state()
    entropy_ghz = metrics_3.entanglement_entropy(ghz_3, [0])
    print(f"   GHZ state entropy (1 vs 2): {entropy_ghz:.6f} bits")
    
    print("\n2. Schmidt Decomposition Analysis:")
    schmidt_coeffs, _, _ = metrics_2.schmidt_decomposition_2qubit(bell_state)
    schmidt_rank = metrics_2.schmidt_rank(bell_state)
    print(f"   Bell state Schmidt coefficients: {schmidt_coeffs}")
    print(f"   Bell state Schmidt rank: {schmidt_rank}")
    
    schmidt_coeffs_prod, _, _ = metrics_2.schmidt_decomposition_2qubit(product_state)
    schmidt_rank_prod = metrics_2.schmidt_rank(product_state)
    print(f"   Product state Schmidt coefficients: {schmidt_coeffs_prod}")
    print(f"   Product state Schmidt rank: {schmidt_rank_prod}")
    
    print("\n3. Concurrence Analysis:")
    conc_bell = metrics_2.concurrence(bell_state)
    conc_product = metrics_2.concurrence(product_state)
    print(f"   Bell state concurrence: {conc_bell:.6f}")
    print(f"   Product state concurrence: {conc_product:.6f}")
    
    print("\n4. Negativity Analysis:")
    neg_bell = metrics_2.negativity(bell_state, [0])
    neg_product = metrics_2.negativity(product_state, [0])
    print(f"   Bell state negativity: {neg_bell:.6f}")
    print(f"   Product state negativity: {neg_product:.6f}")
    
    print("\n5. Multipartite Entanglement Analysis:")
    multipartite_results = metrics_3.multipartite_entanglement_entropy(ghz_3)
    print(f"   GHZ state global entanglement: {multipartite_results['global_entanglement']:.6f}")
    print(f"   GHZ state max bipartite entanglement: {multipartite_results['max_bipartite_entanglement']:.6f}")
    
    # Meyer-Wallach measure
    mw_ghz = metrics_3.meyer_wallach_measure(ghz_3)
    print(f"   GHZ state Meyer-Wallach measure: {mw_ghz:.6f}")


def demonstrate_quantum_verification():
    """Demonstrate quantum state verification capabilities"""
    print("\n" + "=" * 60)
    print("✅ QUANTUM VERIFICATION DEMO")
    print("=" * 60)
    
    gen_2 = QuantumStateGenerator(2)
    
    print("\n1. State Normalization Verification:")
    bell_state = gen_2.bell_state("phi_plus")
    is_normalized = gen_2.verify_normalization(bell_state)
    print(f"   Bell state normalized: {is_normalized}")
    
    # Non-normalized state
    non_normalized = np.array([1, 1, 0, 0], dtype=complex)
    is_normalized_false = gen_2.verify_normalization(non_normalized)
    print(f"   Non-normalized state: {is_normalized_false}")
    
    print("\n2. State Fidelity Analysis:")
    bell_phi_plus = gen_2.bell_state("phi_plus")
    bell_phi_minus = gen_2.bell_state("phi_minus")
    bell_psi_plus = gen_2.bell_state("psi_plus")
    
    fidelity_same = gen_2.state_fidelity(bell_phi_plus, bell_phi_plus)
    fidelity_orthogonal = gen_2.state_fidelity(bell_phi_plus, bell_phi_minus)
    fidelity_different = gen_2.state_fidelity(bell_phi_plus, bell_psi_plus)
    
    print(f"   |Φ+⟩ with itself: {fidelity_same:.6f}")
    print(f"   |Φ+⟩ with |Φ-⟩: {fidelity_orthogonal:.6f}")
    print(f"   |Φ+⟩ with |Ψ+⟩: {fidelity_different:.6f}")
    
    print("\n3. Random State Generation Verification:")
    random_state_1 = gen_2.random_pure_state(seed=42)
    random_state_2 = gen_2.random_pure_state(seed=42)
    random_state_3 = gen_2.random_pure_state(seed=123)
    
    print(f"   Random state 1 normalized: {gen_2.verify_normalization(random_state_1)}")
    print(f"   Same seed reproducible: {np.allclose(random_state_1, random_state_2)}")
    print(f"   Different seeds differ: {not np.allclose(random_state_1, random_state_3)}")


def main():
    """Main demonstration function"""
    print("🎯 ENHANCED ML-PHYS QUANTUM CAPABILITIES DEMONSTRATION")
    print("Showcasing improvements for PhD application portfolio\n")
    
    try:
        demonstrate_quantum_states()
        demonstrate_entanglement_analysis()
        demonstrate_quantum_verification()
        
        print("\n" + "=" * 60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Improvements Achieved:")
        print("✅ Comprehensive quantum state generation")
        print("✅ Rigorous entanglement quantification")
        print("✅ Robust testing framework (38 tests)")
        print("✅ Mathematical correctness verification")
        print("✅ PhD-level theoretical depth")
        
        print("\nNext Steps for PhD Application:")
        print("🔬 Benchmark against classical ML models")
        print("📊 Performance comparison experiments")  
        print("🚀 Quantum resource certification features")
        print("📝 Research paper draft preparation")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())