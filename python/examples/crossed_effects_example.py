#!/usr/bin/env python3
"""
Example: Crossed Random Effects Model using Method of Moments

This example demonstrates the use of the crossed_model() function for fitting
linear mixed models with two crossed random effects. The Method of Moments (MoM)
solver provides O(N) computational complexity, making it ideal for large-scale
datasets.

Scenario: Student Performance Across Schools
--------------------------------------------
We analyze student test scores from a study where students are tested in multiple
schools (a crossed design, as each student visits multiple schools and each school
receives multiple students). This creates two crossed random effects:
  - Student-level variation (some students perform better overall)
  - School-level variation (some schools have better facilities/teachers)

Model: y_ijk = β₀ + β₁x_ijk + u_i + v_j + e_ijk

where:
  - y_ijk is the test score for student i in school j, replicate k
  - x_ijk is a student-level covariate (e.g., study hours)
  - u_i ~ N(0, σ²_student) is the student random effect
  - v_j ~ N(0, σ²_school) is the school random effect
  - e_ijk ~ N(0, σ²_residual) is the residual error
"""

import numpy as np
import pandas as pd
from semx import crossed_model

# Set random seed for reproducibility
np.random.seed(123)

# Simulation parameters
n_students = 50
n_schools = 20
n_reps_per_cell = 3  # Each student visits each school 3 times

# True variance components
true_sigma2_student = 0.6  # Student variance
true_sigma2_school = 0.4   # School variance
true_sigma2_residual = 0.3 # Residual variance

# True fixed effects
true_intercept = 75.0  # Baseline test score
true_slope = 2.5       # Effect of study hours

# Generate random effects
student_effects = np.sqrt(true_sigma2_student) * np.random.randn(n_students)
school_effects = np.sqrt(true_sigma2_school) * np.random.randn(n_schools)

# Generate data
# Key: We use the same student_effect for all observations of that student
# and the same school_effect for all observations in that school
data = []
idx = 0
for student_id in range(n_students):
    for school_id in range(n_schools):
        for rep in range(n_reps_per_cell):
            # Covariate: study hours (standardized)
            study_hours = np.random.randn()

            # True model: y = β₀ + β₁*X + u_student + v_school + e
            score = (true_intercept +
                    true_slope * study_hours +
                    student_effects[student_id] +  # Same for all obs of this student
                    school_effects[school_id] +     # Same for all obs in this school
                    np.sqrt(true_sigma2_residual) * np.random.randn())  # Independent residual

            data.append({
                'student': student_id,
                'school': school_id,
                'study_hours': study_hours,
                'score': score
            })
            idx += 1

df = pd.DataFrame(data)

print("=" * 70)
print("Crossed Random Effects Model Example")
print("=" * 70)
print(f"\nDataset: {len(df)} observations")
print(f"  Students: {df['student'].nunique()}")
print(f"  Schools: {df['school'].nunique()}")
print(f"  Avg observations per student-school pair: {n_reps_per_cell}")
print()

# Fit the model using Method of Moments
print("Fitting model using Method of Moments (O(N) complexity)...")
print()

result = crossed_model(
    formula="score ~ 1 + study_hours",
    data=df,
    u="student",
    v="school",
    use_gls=False,  # Use OLS for initial fixed effects
    second_step=False,  # Single-step estimation
    verbose=False
)

print(result)
print()

# Compare estimates to true values
print("=" * 70)
print("Comparison to True Values")
print("=" * 70)
print()
print("Fixed Effects:")
print(f"  Intercept:    True = {true_intercept:6.2f},  Estimated = {result.beta[0]:6.2f}")
print(f"  Slope:        True = {true_slope:6.2f},  Estimated = {result.beta[1]:6.2f}")
print()
print("Variance Components:")
print(f"  σ²_student:   True = {true_sigma2_student:6.4f},  Estimated = {result.variance_components[0]:6.4f}")
print(f"  σ²_school:    True = {true_sigma2_school:6.4f},  Estimated = {result.variance_components[1]:6.4f}")
print(f"  σ²_residual:  True = {true_sigma2_residual:6.4f},  Estimated = {result.variance_components[2]:6.4f}")
print()

# Compute relative errors
rel_error_student = abs(result.variance_components[0] - true_sigma2_student) / true_sigma2_student
rel_error_school = abs(result.variance_components[1] - true_sigma2_school) / true_sigma2_school
rel_error_residual = abs(result.variance_components[2] - true_sigma2_residual) / true_sigma2_residual

print("Relative Errors:")
print(f"  Student:   {rel_error_student*100:5.1f}%")
print(f"  School:    {rel_error_school*100:5.1f}%")
print(f"  Residual:  {rel_error_residual*100:5.1f}%")
print()

# Demonstrate GLS refinement
print("=" * 70)
print("Comparison: OLS vs GLS Refinement")
print("=" * 70)
print()

result_gls = crossed_model(
    formula="score ~ 1 + study_hours",
    data=df,
    u="student",
    v="school",
    use_gls=True,  # Use GLS for refined fixed effects
    second_step=False
)

print(f"Fixed Effects (OLS):     β₀ = {result.beta[0]:6.2f}, β₁ = {result.beta[1]:6.2f}")
print(f"Fixed Effects (GLS):     β₀ = {result_gls.beta[0]:6.2f}, β₁ = {result_gls.beta[1]:6.2f}")
print(f"True Values:             β₀ = {true_intercept:6.2f}, β₁ = {true_slope:6.2f}")
print()

print("=" * 70)
print("Key Advantages of Method of Moments")
print("=" * 70)
print()
print("1. Computational Efficiency:")
print("   - O(N) complexity vs O(N³) for ML/REML")
print("   - Scales to millions of observations")
print()
print("2. No Convergence Issues:")
print("   - Direct solver, no iterative optimization")
print("   - Always provides estimates (no convergence failures)")
print()
print("3. Ideal for Large-Scale Applications:")
print("   - Genomic prediction with millions of SNPs")
print("   - Educational data with many students and schools")
print("   - Industrial quality control with crossed factors")
print()
print("Limitations:")
print("- Restricted to linear models with Gaussian residuals")
print("- Requires two (and only two) crossed random effects")
print("- May have higher variance than ML/REML for small samples")
print()
