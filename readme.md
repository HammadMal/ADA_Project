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
│-- 📂 checkpoint1            # Contains project proposal and notes
│   │-- 📄 Project_Proposal_ADA.pdf
│   │-- 📄 main.tex
│-- 📂 code                   # Source code for implementations
│-- 📂 documentation          # Additional project documentation
│-- 📂 report                 # Reports and writeups
│-- 📂 research_materials     # Paper PDFs, notes, and literature review
│-- 📄 readme.md              # Project description, setup, and instructions


## Implementation Plan
We will implement and test the new APASP algorithms, comparing them against baseline shortest path algorithms to evaluate performance in different graph settings (dense and sparse graphs).

### Key Components:
1. **New APASP Algorithm**
   - Implementing the algorithm as per the paper’s pseudocode
   - Optimizing performance for different graph types
2. **Baseline Comparisons**
   - Implementing traditional All-Pairs Shortest Path (APSP) algorithms
   - Comparing time complexity and accuracy
3. **Experimental Analysis**
   - Running tests on synthetic and real-world graph datasets
   - Evaluating efficiency and approximation accuracy



### Core Concepts from the Paper:
- **Multiplicative 2-approximation:** Ensures the estimated distance between any two vertices is at most twice the true shortest path.
- **Algorithmic Improvements:** The paper introduces two new algorithms:
  - **Faster algorithm** for computing multiplicative 2-approximations with time complexity Õ(min{n1/2m, n9/4}).
  - **Distance-sensitive algorithm** for vertex pairs with distances ≥ 4, achieving Õ(min{n7/4m1/4, n11/5}) expected time.


   These improvements provide better efficiency compared to previous methods, such as those based on Fast Matrix Multiplication (FMM).


### Current Understanding
1. **Problem Context:**
   - The APASP problem seeks to efficiently compute approximate shortest paths between all pairs of vertices in a graph.
   - While exact All Pairs Shortest Paths (APSP) has O(n³) time complexity, approximate solutions can be significantly faster.

2. **Key Algorithm Components:**
   - **Vertex Hierarchies:** Strategic partitioning of vertices into sets V₁, V₂, etc. based on degree thresholds
   - **Edge Hierarchies:** Classification of edges based on their endpoints' inclusion in vertex sets
   - **Blocking Vertices:** Identifying vertices that must be included in certain shortest paths
   - **Case Analysis:** Different handling based on path structures (C1, C2, C3)

3. **The Two Main Algorithms:**
   - **Multiplicative 2-Approximation:** Õ(min{n¹/²m, n⁹/⁴}) time complexity
   - **Distance-Sensitive Algorithm:** Õ(min{n⁷/⁴m¹/⁴, n¹¹/⁵}) expected time for distances ≥ 4

### Implementation Progress
- [x] Basic graph data structures
- [x] Baseline algorithms (Floyd-Warshall, Dijkstra's)
- [x] Core vertex and edge classification functions
- [ ] Case analysis implementation (C1, C2, C3)
- [ ] Post-processing for multiplicative approximation
- [ ] Distance-sensitive optimizations
- [ ] Comprehensive testing framework

## Technical Challenges & Solutions
We've identified several key challenges which we can face during implementation:

1. **Memory Management:** The O(n²) space requirement is demanding for large graphs. We're implementing sparse matrix representations to address this.

2. **Sampling Accuracy:** The probabilistic vertex sampling requires careful tuning. We're working on calibration methods to ensure consistent results.

3. **Algorithm Correctness:** The complex case analysis requires rigorous testing. We've designed a test suite with edge cases to verify correctness.

4. **Performance Optimization:** We're implementing parallel processing for the multiple Dijkstra runs to improve performance.

## Experimental Setup
We will test  our implementations on various graph types:

- **Synthetic Graphs:** 
  - Random Erdős-Rényi graphs with varying densities
  - Small-world networks
  - Scale-free networks

- **Real-world Networks:**
  - Social network datasets
  - Transportation networks
  - Web graphs

Our benchmarks will measure both runtime efficiency and approximation quality across different graph sizes and densities.

## Findings & Insights (Preliminary)
- The new algorithms show promising results for dense graphs, with significant speedups compared to traditional APSP approaches
- The case analysis approach successfully handles different path structures within the graph
- For sparse graphs, the multiplicative approximation algorithm approaches the efficiency of BFS-based methods


## Team Members & Responsibilities
| Name           | Role & Responsibilities |
|---------------|-------------------------|
| Hammad Malik | Implementing new APASP algorithms, time complexity analysis |
| Mahrukh Yousuf | Conducting experiments, comparing results with baselines, documentation |

## Development Setup
### Prerequisites
- Python 3.x
- Required libraries: `networkx`, `numpy`, `matplotlib`
- LaTeX for report writing

### Setup Instructions

   ```

## Contribution Guidelines
- Follow best coding practices and keep commits descriptive.
- Maintain a clean repository (no unnecessary files like backup, autosave, metadata).
- Create feature branches for major changes before merging to `master`.

## References
- Roditty, L. (2023). *New Algorithms for All Pairs Approximate Shortest Paths*. STOC 2023.

