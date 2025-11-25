from src.two_layer_routing import (
    chain_decomposition_matching,
    verify_chain_decomposition,
)

# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Chain Decomposition using Ford-Fulkerson Algorithm")
    print("=" * 70)

    # Example 1
    print("\nExample 1: Simple permutations")
    perm1 = [2, 0, 1, 3]
    perm2 = [1, 2, 0, 3]

    print(f"Permutation 1: {perm1}")
    print(f"Permutation 2: {perm2}")

    matching, chains, max_flow = chain_decomposition_matching(perm1, perm2)

    print(f"\nMaximum flow (matching size): {max_flow}")
    print(f"Matching edges: {matching}")
    print(f"Number of chains: {len(chains)}")
    print(f"Chain decomposition:")
    for i, chain in enumerate(chains):
        print(f"  Chain {i+1}: {' → '.join(map(str, chain))}")

    valid, msg = verify_chain_decomposition(perm1, perm2, chains)
    print(f"\nVerification: {msg}")

    # Example 2
    print("\n" + "=" * 70)
    print("Example 2: Identity permutations")
    perm1 = [0, 1, 2, 3, 4]
    perm2 = [0, 1, 2, 3, 4]

    print(f"Permutation 1: {perm1}")
    print(f"Permutation 2: {perm2}")

    matching, chains, max_flow = chain_decomposition_matching(perm1, perm2)

    print(f"\nMaximum flow: {max_flow}")
    print(f"Matching edges: {matching}")
    print(f"Number of chains: {len(chains)}")
    print(f"Chain decomposition:")
    for i, chain in enumerate(chains):
        print(f"  Chain {i+1}: {' → '.join(map(str, chain))}")

    valid, msg = verify_chain_decomposition(perm1, perm2, chains)
    print(f"\nVerification: {msg}")

    # Example 3
    print("\n" + "=" * 70)
    print("Example 3: Reversed permutations")
    perm1 = [0, 1, 2, 3, 4]
    perm2 = [4, 3, 2, 1, 0]

    print(f"Permutation 1: {perm1}")
    print(f"Permutation 2: {perm2}")

    matching, chains, max_flow = chain_decomposition_matching(perm1, perm2)

    print(f"\nMaximum flow: {max_flow}")
    print(f"Matching edges: {matching}")
    print(f"Number of chains: {len(chains)}")
    print(f"Chain decomposition:")
    for i, chain in enumerate(chains):
        print(f"  Chain {i+1}: {' → '.join(map(str, chain))}")

    valid, msg = verify_chain_decomposition(perm1, perm2, chains)
    print(f"\nVerification: {msg}")

    # Example 4: Longer chain
    print("\n" + "=" * 70)
    print("Example 4: Permutations with longer chains")
    perm1 = [0, 1, 2, 3, 4, 5]
    perm2 = [1, 0, 3, 2, 5, 4]

    print(f"Permutation 1: {perm1}")
    print(f"Permutation 2: {perm2}")

    matching, chains, max_flow = chain_decomposition_matching(perm1, perm2)

    print(f"\nMaximum flow: {max_flow}")
    print(f"Matching edges: {matching}")
    print(f"Number of chains: {len(chains)}")
    print(f"Chain decomposition:")
    for i, chain in enumerate(chains):
        print(f"  Chain {i+1}: {' → '.join(map(str, chain))}")

    valid, msg = verify_chain_decomposition(perm1, perm2, chains)
    print(f"\nVerification: {msg}")

    print("\n" + "=" * 70)
