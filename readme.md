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
📂 APASP-Project
│-- 📄 README.md  # Project description, setup, and instructions
│-- 📂 research   # Paper PDFs, notes, and literature review
│-- 📂 code       # Source code for implementations
│-- 📂 reports    # Reports and documentation
│   │-- checkpoint1/ # Contains LaTeX source & PDF for Checkpoint 1
│-- 📂 docs       # Additional project documentation
│-- .gitignore    # Specifies files to exclude from version control
```

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
- Create feature branches for major changes before merging to `main`.

## References
- Roditty, L. (2023). *New Algorithms for All Pairs Approximate Shortest Paths*. STOC 2023.

