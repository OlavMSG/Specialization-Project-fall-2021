# TMA4500 Industrial Mathematics, Specialization Project

My code for building a reduced order model for the linear elasticity equation on a square in 2D.

Please see the [Code used in report](code_use_for_report) for examples and plotting results 
from the report, as the [Patch Test](code_use_for_report/Patch_Test), 
[Example 1 traction forces](code_use_for_report/Example_1_traction_forces) and 
[Example 2 ``Gravity in 2D´´](code_use_for_report/Example_2_Gravity_in_2D).
Whereas [Code not use in report](code_not_used_for_report) has some unused example code and 
old plotting/example code.

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



