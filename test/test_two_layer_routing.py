import os
import sys

# Ensure repository root is on sys.path so `src` is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rus import (
    chain_decomposition_matching,
    verify_chain_decomposition,
)


# Example usage
def test_chain_decomposition():
    # print("=" * 70)
    # print("Chain Decomposition using Ford-Fulkerson Algorithm")
    # print("=" * 70)

    # Example 1
    # print("\nExample 1: Simple permutations")
    perm1 = [2, 0, 1, 3]
    perm2 = [1, 2, 0, 3]

    # print(f"Permutation 1: {perm1}")
    # print(f"Permutation 2: {perm2}")

    matching, chains = chain_decomposition_matching(perm1, perm2)

    # print(f"Matching edges: {matching}")
    # print(f"Number of chains: {len(chains)}")
    # print("Chain decomposition:")
    # for i, chain in enumerate(chains):
    #     print(f"  Chain {i+1}: {' → '.join(map(str, chain))}")

    valid, msg = verify_chain_decomposition(perm1, perm2, chains)
    assert valid
    # Example 2
    # print("Example 2: Identity permutations")
    perm1 = [0, 1, 2, 3, 4]
    perm2 = [0, 1, 2, 3, 4]

    # print(f"Permutation 1: {perm1}")
    # print(f"Permutation 2: {perm2}")

    matching, chains = chain_decomposition_matching(perm1, perm2)

    # print(f"Matching edges: {matching}")
    # print(f"Number of chains: {len(chains)}")
    # print("Chain decomposition:")
    # for i, chain in enumerate(chains):
    #     print(f"  Chain {i+1}: {' → '.join(map(str, chain))}")

    valid, msg = verify_chain_decomposition(perm1, perm2, chains)
    # print(f"\nVerification: {msg}")
    assert valid

    # Example 4: Longer chain
    print("\n" + "=" * 70)
    print("Example 4: Permutations with longer chains")
    perm1 = [0, 1, 2, 3, 4, 5]
    perm2 = [1, 0, 3, 2, 5, 4]

    # print(f"Permutation 1: {perm1}")
    # print(f"Permutation 2: {perm2}")

    matching, chains = chain_decomposition_matching(perm1, perm2)

    # print(f"Matching edges: {matching}")
    # print(f"Number of chains: {len(chains)}")
    # print("Chain decomposition:")
    # for i, chain in enumerate(chains):
    #     print(f"  Chain {i+1}: {' → '.join(map(str, chain))}")

    valid, msg = verify_chain_decomposition(perm1, perm2, chains)
    assert valid
