# Code: TMA4500 Industrial Mathematics, Specialization Project

My code for building a reduced-order model for the linear elasticity equation on a square in 2D, solving for plane stress, under the conditons that the body force f, the prescribed traction force(s) h on the Neumann bondary and the prescribed displacment u on the Dirichlet boundary do not depend on my choice of parameters, the Young's module E and the Poisson ration &nu;.

Please see the [Code used in report](code_use_for_report) for code for the examples and the plotting results 
from the report, as the [Patch Test](code_use_for_report/Patch_Test), 
[Example 1 traction forces](code_use_for_report/Example_1_traction_forces) and 
[Example 2 ``Gravity in 2D´´](code_use_for_report/Example_2_Gravity_in_2D).

[Code not use in report](code_not_used_for_report) has some unused code (example code) and 
old plotting/example code.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5786534.svg)](https://doi.org/10.5281/zenodo.5786534)

## The solver Class
Please see [LinearElasticity2DProblem](src/linear_elasticity_2d_solver/_linear_elasticity_2d_problem_class.py)
for more documentation.

### Default constants
Please see [Default constants](src/linear_elasticity_2d_solver/default_constants.py)
for documentation.

### Useful helper functions
Please see [Helpers](src/linear_elasticity_2d_solver/helpers.py)
for documentation.

### Exceptions
Please see [Exceptions](src/linear_elasticity_2d_solver/exceptions.py)
for documentation.

### Triangulation
Please see [Get plate](src/linear_elasticity_2d_solver/get_plate.py)
for documentation.

### Other documentation
#### High-fidelity assembly
Please see [Default constants](src/linear_elasticity_2d_solver/default_constants.py)
for documentation.

#### Gauss quadrature
Please see [Gauss quadrature](src/linear_elasticity_2d_solver/_gauss_quadrature.py)
for documentation.

#### Plotting
Please see [Plotting](src/linear_elasticity_2d_solver/_plotting.py)
for documentation.

#### Proper orthogonal decomposition with respect to the energy norm
Please see [POD](src/linear_elasticity_2d_solver/_pod.py)
for documentation.

#### Reduced-order data class
Please see [RB data](src/linear_elasticity_2d_solver/_rb_data_class.py)
for documentation.

#### Saving and loading matrices and vectors form files
Please see [Save and load](src/linear_elasticity_2d_solver/_save_and_load.py)
for documentation.

#### Solution Function class
Please see [Solution function class](src/linear_elasticity_2d_solver/_solution_function_class.py)
for documentation.

#### Stress recovery
Please see [Stress recovery](src/linear_elasticity_2d_solver/_stress_recovery.py)
for documentation.



