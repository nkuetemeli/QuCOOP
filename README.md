# QuCOOP: Composite Optimization on Quantum Annealers
This code can be used to reproduce the experiments and results of the paper <br/>
**A Framework for Solving Composite and Binary-Parametrised Problems on Quantum Annealers**.

    Authors 
    A Framework for Solving Composite and Binary-Parametrised Problems on Quantum Annealers. 
    Publication Venur. URL

QuCOOP is a general framework for solving, on quantum annealers, problems of the form

$$min_{x\in \left\{0, 1\right\}^n} \quad f(g(x)),$$

where $f$ is quadratic in $g(x)$.

This repository demonstrates the performance of QuCOOP on two problems:
- Given a symmetric matrix as well as an alpha penalty factor, the `QAP` class in `src/qap` finds a locally optimal permutation matrix that minimizes the quadratic assignment problem.
- Given a reference and template point sets as well as alpha and beta penalty factor, the `PointSetRegistration` class in `src/psr` performs a rigid registration of the point sets without correspondences known in advance.

# Install
The code depends on the Python packages 
[numpy](https://numpy.org/install/), 
[dwave](https://docs.ocean.dwavesys.com/projects/system/en/latest/installation.html),
and (for benchmarks and plots)
[pycpd](https://pypi.org/project/pycpd/), 
[trimesh](https://pypi.org/project/trimesh/),
[matplotlib](https://pypi.org/project/matplotlib/).

- Please download the repository and install the requirements in `requirements.txt` or refer to the product pages for reference.

- Once you satisfied the dependency, run `python -m pip install .` inside the directory.

Move to the `src` folder to run the subsequent commands.

# Example

    # Run tiny random porblems br running `psr.py` and `qap.py` directly
    # Close the Plots as they pop-up to continue the execution of the script
    python psr.py
    python qap.py

# Run demos

    # Demos are pre-fixed by the word 'demo_' 
    # Try some demos, specify while calling the solver wheter to use simulated or quantum annealing
    python demo_point_set_registration.py
    python demo_quadratic_assignment.py
    python demo_shape_matching.py

# Citation
If you find this work useful, please cite the article [Article URL](#).
