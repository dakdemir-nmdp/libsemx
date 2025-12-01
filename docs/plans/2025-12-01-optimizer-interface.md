# Plan: Optimizer Plug-in Interface (2025-12-01)

- Review blueprint §§2.7 and 3.8 to capture optimizer requirements (pluggable L-BFGS/Newton, analytic gradients, convergence stats).
- Introduce base `ObjectiveFunction`, `Optimizer`, and result/options structs to standardize interactions.
- Provide a simple gradient-descent optimizer stub to exercise the interface until advanced solvers land.
- Cover iteration limits, tolerance checks, and option validation with Catch2 tests on a quadratic objective.
