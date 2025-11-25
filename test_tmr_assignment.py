from src.tmr_assignment import MagicStateMatching


# Example usage
if __name__ == "__main__":
    # Example 1
    print("EXAMPLE 1")
    M1 = [4, 3, 2, 5, 1]
    A1 = [5, 4, 4, 2]

    matcher1 = MagicStateMatching(M1, A1)
    matcher1.run()
    matcher1.print_results()

    print("\n" + "=" * 80)
    print("\n")

    # Example 2
    # print("EXAMPLE 2")
    M2 = [10, 5, 4, 6, 8, 1]
    A2 = [10, 5, 3, 7, 9]

    matcher2 = MagicStateMatching(M2, A2)
    matcher2.run()
    matcher2.print_results()
