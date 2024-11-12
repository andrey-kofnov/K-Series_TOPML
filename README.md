# K-Series_TOPML
## Moment-based Density Elicitation with Applications in Probabilistic Loops

This repository contains supplementary material (data and source code) to support the TOPML paper submission titled **"Moment-based Density Elicitation with Applications in Probabilistic Loops."**

---

## Description of Files

- **`requirements.txt`**: Python requirements file for installing dependencies to execute `main.py`.
- **`main_solutions.py`**: Script to run benchmarks as shown in Table 1 of the paper.
- **`K_series_computation.py`**: Script containing all mathematical logic as described in the paper.
- **`ort_poly2.py`**: Python code for Polynomial Chaos Expansion.
- **`utils.py`**: Contains utility functions and definitions of distributions.
- **`S1Dataset.txt`**: Dataset from Munkhammar et al. (2017).
- **`Sampling.py`**: Script used by `main_solutions.py` to generate samples for benchmarks.
- **`Problems.py`**: Definitions of benchmark problems and their parameters.

## Installation and Usage

1. **Set up the environment**: Install required packages by running:
   ```bash
   pip install -r requirements.txt

(Note: The code was written for Python 3.8.)

2. **Run a Benchmark Problem**: Use the command line to run a specific problem:

   ```bash
   python main_solutions.py 'Problem_Name'

Alternatively, you can add an optional easy_mode argument:

   ```bash
   python main_solutions.py 'Problem_Name' 'easy_mode'









