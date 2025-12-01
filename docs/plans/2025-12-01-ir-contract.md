# Plan: Front-End IR Contract (2025-12-01)

- Review blueprint §§2.1 and 10 to extract core entities (variables, edges, random effects, covariance modules, outcome families).
- Define C++ `ModelIR` structs mirroring the shared schema plus a builder that validates names and references.
- Provide room for future extensions via string identifiers (families, covariance IDs) while covering SEM and mixed-model essentials.
- Exercise invariants in Catch2 tests (duplicate variables, unknown references, successful assembly) to anchor future Python/R serialization.
