# `TSPUtils` - Utilities for Symmetric TSP Research

Utilities and tools for working with **symmetric Traveling Salesman Problem (TSP)** instances and solvers.

This repository provides convenient interfaces and helper functions for parsing, manipulating, benchmarking, and visualizing TSP instances, with a focus on supporting experimental research workflows (e.g., comparing solvers such as Concorde and LKH3, evaluating runtimes, generating features, etc.).

---

## 🚀 Features

`tsp_utils` offers functionalities including:

- Loading and parsing TSPLIB-formatted instances
- Utilities for running and benchmarking TSP solvers
  - Wrappers for common solvers
  - Log parsing and runtime extraction

Several function are built using the `tsplib95` library, which provides a convenient interface 
for working with TSP instances in Python. The [`tsplib95`](https://pypi.org/project/tsplib95/) library allows you to easily load and 
manipulate TSP instances, and it provides a simple API for accessing the data contained in the 
TSPLIB files. However, for us, an **instance** is a `nx.Graph` object. 


---

## ✨ Main methods
Among other methods, you have:
- `stsp.solve_tsp` with `pySCIPopt`
- `stsp.solve_tsp_fixed_edge`, which allows to solve a TSP instance with a fixed edge (i.e., a partial solution) using `pySCIPopt`. 
- `stsp.run_concorde` that runs Concorde on a given TSP instance and returns the solution, runtime, and other things
- `stsp.run_concorde_LB` that just runs the linear relaxation (Dantzig-Fulkerson-Johnson formulation) of Concorde and returns the lower bound and the runtime
- `stsp.solve_sep` that solves the linear relaxation with pySCIPopt
- `utils.create_tsp_problem_object` which creates a `tsplib95.tsp.TSPProblem` object
- `utils.read_tsplib` that reads a TSPLIB instance and returns a `nx.Graph` or `nx.DiGraph` object
- `tsp.run_LKH` that runs LKH3 on a given TSP instance and returns the solution, runtime, and other things
---
## 🆘 Installation
Just clone this repository where you want to use it and install the package using pip:

```bash
cd tsp_utils
python -m pip install -e .
``` 
---
## 📁 Repository Structure

```text
tsp_utils/
├── tsp_utils/              # Python package
│   ├── stsp.py             # Utilities for symmetric TSP instances
    ├── tsp.py              # Utilities for all TSP types (including asymmetric)
    ├── utils.py            # General helper functions, mostly for parsing and loading instances
└── README.md
```
---
## 🧑‍💻 Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.
Or, if you really don't want to code but have suggestions, you can also just send me an email at `eleonora.vercesi@usi.ch`

---
## ⚠️ Warning 

`pySCIPopt` does not work in the `base` conda environment, so you need to create a new conda environment and install `pySCIPopt` there.

---
## 📅 Last update
This is a work in progress, and it is constantly being updated and improved. The last update was on

**February 2026**