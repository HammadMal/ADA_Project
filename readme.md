# All Pairs Approximate Shortest Paths (APASP) Project

## Project Overview
This project focuses on implementing and analyzing new algorithms for the **All Pairs Approximate Shortest Paths (APASP)** problem in unweighted, undirected graphs. The selected paper introduces improved algorithms for APASP that do not rely on Fast Matrix Multiplication (FMM), targeting better time complexity for dense graphs.

### Paper Details
- **Title:** New Algorithms for All Pairs Approximate Shortest Paths
- **Author:** Liam Roditty
- **Conference:** 55th Annual ACM Symposium on Theory of Computing (STOC 2023)
- **DOI/Link:** [ACM Digital Library](https://doi.org/10.1145/3564246.3585197)

## Repository Structure
```
📂 ADA_Project
│-- 📂 checkpoint1                 # Contains project proposal and notes.
│-- 📂 checkpoint2                 # Contains checkpoint2 pdf and tex.
│-- 📂 checkpoint3                 # Contains checkpoint3 pdf and tex.
│   │-- 📄 CP3_Progress_Report.pdf # Checkpoint 3 Progress Report. 
│   │-- 📄 main.tex                # Tex file.

│-- 📂 checkpoint4                 # Contains checkpoint4 pdf and tex.
│   │-- 📄 Final_Report.pdf        # Final Report. 
│   │-- 📄 main.tex                # Tex file.

│-- 📂 code                        # Source code for implementations
│   │-- 📄 implementation.py       # Main implementation of algorithms

│-- 📂 documentation               # Additional project documentation
│   │-- 📄 readme.md               # Code Documentation

│-- 📂 research_materials          # Paper PDFs, notes, and literature review
│-- 📂 Results                     # Results such as running times, speedups and comparisions.

│-- 📄 readme.md                   # Project description, setup, and instructions
```

## Implementation Plan
We have implemented and tested the new APASP algorithms, comparing them against baseline shortest path algorithms to evaluate performance in different graph settings (dense and sparse graphs).

### Key Components:
1. **New APASP Algorithm**
   - Implemented the algorithm as per the paper's pseudocode
   - Optimized performance for different graph types
   - Created specialized test graph generators to showcase algorithm properties
2. **Baseline Comparisons**
   - Implemented Floyd-Warshall as the baseline exact APSP algorithm
   - Compared time complexity and accuracy with detailed metrics
3. **Experimental Analysis**
   - Conducted tests on synthetic graphs of various sizes and densities
   - Evaluated efficiency and approximation accuracy with comprehensive metrics

### Core Concepts from the Paper:
- **Multiplicative 2-approximation:** Ensures the estimated distance between any two vertices is at most twice the true shortest path.
- **Algorithmic Improvements:** The paper introduces two new algorithms:
  - **Faster algorithm** for computing multiplicative 2-approximations with time complexity Õ(min{n^(1/2)m, n^(9/4)}).
  - **Distance-sensitive algorithm** for vertex pairs with distances ≥ 4, achieving Õ(min{n^(7/4)m^(1/4), n^(11/5)}) expected time.

These improvements provide better efficiency compared to previous methods, such as those based on Fast Matrix Multiplication (FMM).

## Implementation Progress
- [x] Basic graph data structures
- [x] Baseline algorithms (Floyd-Warshall, Dijkstra's)
- [x] Core vertex and edge classification functions
- [x] apasp_k algorithm implementation
- [x] Case analysis implementation (C1, C2, C3)
- [x] Post-processing for multiplicative approximation
- [x] Comprehensive testing framework
- [x] Performance analysis and comparison

## Latest Results
Our implementation has yielded several key findings:

### Key Findings
1. The speedup factor increases with graph size, reaching approximately 8x for graphs with 200 vertices
2. The approximation algorithm clearly demonstrates the theoretical time complexity advantages
3. Current implementation occasionally exceeds the theoretical multiplicative-2 bound in certain test cases
4. The algorithm performs better on sparse graphs than dense graphs in terms of approximation quality

## Technical Challenges & Solutions
We've addressed several key challenges during implementation:

1. **Hitting Set Construction:** We implemented a hybrid approach combining deterministic and probabilistic methods to balance accuracy and performance.

2. **Zero Error Issue:** Our initial implementation computed exact distances rather than approximations. We addressed this by introducing appropriate approximation for longer distances.

3. **Test Graph Generation:** We created specialized test graph generators that ensure connectivity and create structures with long paths to better demonstrate the algorithm's behavior.

4. **Approximation Guarantee:** We identified issues with maintaining the theoretical multiplicative-2 bound and are working on refining the implementation to strictly adhere to this guarantee.

## Experimental Setup
We tested our implementations on various graph types:

- **Synthetic Graphs:** 
  - Sparse graphs with long paths
  - Dense graphs with clear cluster structures
  - Connected random graphs with varying edge probabilities

Our benchmarks measure both runtime efficiency and approximation quality across different graph sizes, with multiple trials to ensure stable results.

## Team Members & Responsibilities
| Name           | Role & Responsibilities |
|---------------|-------------------------|
| Hammad Malik | Implementing new APASP algorithms, time complexity analysis |
| Mahrukh Yousuf | Conducting experiments, comparing results with baselines, documentation |

## Development Setup
### Prerequisites
- Python 3.6+
- Required libraries: `numpy`, `matplotlib`

### Setup Instructions
1. Clone the repository
2. Install required packages: `pip install numpy matplotlib`
3. Run the implementation: `python implementation.py`

## Future Work
1. **Strict Approximation Guarantees:** Refine the implementation to ensure the multiplicative-2 bound is never exceeded
2. **Large-Scale Testing:** Extend testing to larger graphs to further demonstrate algorithmic advantages
3. **Real-World Networks:** Apply the algorithm to real-world network datasets
4. **Parallel Implementation:** Explore parallel processing for further performance improvements

## References
- Roditty, L. (2023). *New Algorithms for All Pairs Approximate Shortest Paths*. STOC 2023.