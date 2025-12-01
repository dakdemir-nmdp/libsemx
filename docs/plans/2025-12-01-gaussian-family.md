# Plan: Gaussian Outcome Family (2025-12-01)

- Revisit blueprint §§3.6 and 6.1 to confirm outcome API expectations (loglik, gradient, hessian hooks).
- Introduce abstract `OutcomeFamily` interface plus Gaussian specialization providing analytic derivatives.
- Wire Gaussian unit tests validating log-likelihood, gradient, and Hessian against closed-form results.
- Update CMake/test harness and TODO log; ensure future families can plug into same interface.
