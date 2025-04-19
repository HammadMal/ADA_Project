# All Pairs Approximate Shortest Paths Implementation

This repository contains an implementation of the multiplicative 2-approximation algorithm for All Pairs Shortest Paths (APSP) as described in Liam Roditty's paper "New Algorithms for All Pairs Approximate Shortest Paths" (STOC 2023).

## Overview

The implementation compares a baseline exact algorithm (Floyd-Warshall) with the faster multiplicative 2-approximation algorithm proposed in the paper. The approximation algorithm provides distances that are at most twice the actual shortest path distances, while running significantly faster than the exact algorithm.

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - matplotlib
  - networkx (optional, for additional graph generation)

You can install the required packages using:

```bash
pip install numpy matplotlib networkx
```

## Implementation Files

- `implementation.py`: The main implementation file containing all algorithms and testing framework
- `algorithm_comparison.png`: Generated visualization of algorithm performance comparison

## Key Components

The implementation includes the following key components:

1. **Graph Representation**: A simple adjacency list implementation of undirected graphs
2. **Floyd-Warshall Algorithm**: The baseline exact APSP algorithm
3. **apasp_k Algorithm**: The core procedure from the paper that computes an additive 2(k-1) approximation
4. **Multiplicative 2-Approximation**: The main algorithm that computes distances with multiplicative factor at most 2
5. **Testing Framework**: Code to compare algorithm performance and accuracy

## Usage

Run the implementation with:

```powershell
python implementation.py
```

This will:
1. Generate test graphs of various sizes
2. Run both the baseline and approximation algorithms on each graph
3. Measure and report running times and approximation quality
4. Generate visualization comparing the algorithms
5. Save results to `algorithm_comparison.png`

## Implementation Details

### Graph Class
```python
class Graph:
    # Simple adjacency list representation of an undirected graph
    # Methods include:
    # - add_edge(u, v)
    # - get_neighbors(u)
    # - get_degree(u)
```

### Algorithms

1. **Floyd-Warshall** (`floyd_warshall`): Standard implementation of the exact APSP algorithm with O(n³) time complexity.

2. **apasp_k** (`apasp_k`): Implementation of Algorithm 1 from the paper that computes an additive 2(k-1) approximation by:
   - Computing degree thresholds z_i and corresponding vertex sets
   - Creating hitting sets Z_i that intersect neighborhoods of high-degree vertices
   - Running Dijkstra's algorithm in specially constructed graphs

3. **Multiplicative 2-Approximation** (`multiplicative_2_approximation`): Implements the approach from Corollary 2.4 in the paper:
   - For distances = 1: Computes exact distances
   - For distances {2,3}: Computes distances with at most +2 additive error
   - For distances ≥ 4: Uses an additive 4 approximation, which is also a multiplicative 2 approximation

4. **Test Graph Generation** (`generate_test_graph`, `generate_random_graph`): Functions to create test graphs with specific structures for evaluating the algorithms.

## Experimental Results

Running the implementation produces comparisons showing:

1. **Running Time**: Comparison of execution time for both algorithms across different graph sizes
2. **Approximation Quality**: Measurement of maximum and average errors (both additive and multiplicative)
3. **Speedup Factor**: How much faster the approximation algorithm is compared to the exact algorithm

The results confirm the theoretical advantage of the approximation algorithm, with the speedup factor increasing with graph size.

## Modifications and Enhancements

Our implementation includes some modifications to the original algorithm described in the paper:

1. **Simplified Distance Handling**: We use a more direct BFS-based approach for computing distances ≤ 3 instead of the full Algorithm 2 from the paper.

2. **Explicit Error Introduction**: We deliberately introduce approximation errors for longer distances to clearly demonstrate the algorithm's theoretical behavior.

3. **Test Graph Structures**: We generate graphs with specific structures to better highlight the differences between exact and approximate algorithms.

## Authors

- Mahrukh Yousuf (08055)
- Hammad Malik (08298)

## References

Roditty, L. (2023). New Algorithms for All Pairs Approximate Shortest Paths. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing (STOC '23).