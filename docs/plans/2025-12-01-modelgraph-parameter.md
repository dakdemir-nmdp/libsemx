# Plan: ModelGraph & Parameter Catalog (2025-12-01)

- Revisit blueprint §§2.2–3.2 to confirm the required metadata (variables, regressions, loadings, covariances, parameter constraints) and ensure ModelGraph is the single source of truth that can emit ModelIR for bindings.
- Introduce a `ParameterCatalog` owning `Parameter` objects plus their transforms, providing ordered views (initial constrained/unconstrained values, names, accessors) so `LikelihoodDriver::fit` and gradient code stop re-deriving parameter vectors.
- Rewire `ModelGraph`/`ModelIRBuilder` usage so regressions, loadings, and covariance specs register through the graph, then expose helpers to serialize into ModelIR for the driver/bindings.
- Update Catch2 + Python/R tests to assert duplicate detection, deterministic parameter ordering, and gradient alignment using the new catalog-powered infrastructure.
