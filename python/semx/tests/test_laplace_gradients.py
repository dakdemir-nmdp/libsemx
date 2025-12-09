"""Integration tests that exercise Laplace gradients through the Python bindings."""

from __future__ import annotations

import pytest

semx = pytest.importorskip("semx")


def _build_binomial_model() -> "semx.ModelIR":
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "binomial")
    builder.add_variable("x", semx.VariableKind.Exogenous, "")
    builder.add_variable("cluster", semx.VariableKind.Grouping)

    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")

    builder.add_covariance("G", "diagonal", 1)
    builder.add_random_effect("u", ["cluster"], "G")

    return builder.build()


def _build_negative_binomial_model() -> "semx.ModelIR":
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "negative_binomial")
    builder.add_variable("x", semx.VariableKind.Exogenous, "")
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")
    builder.add_covariance("G_nb", "diagonal", 1)
    builder.add_random_effect("u_cluster", ["cluster"], "G_nb")
    return builder.build()


def _build_kronecker_model() -> "semx.ModelIR":
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "binomial")
    builder.add_variable("x", semx.VariableKind.Exogenous, "")
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_variable("t1e1", semx.VariableKind.Exogenous, "")
    builder.add_variable("t1e2", semx.VariableKind.Exogenous, "")
    builder.add_variable("t2e1", semx.VariableKind.Exogenous, "")
    builder.add_variable("t2e2", semx.VariableKind.Exogenous, "")
    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")
    builder.add_covariance("G_kron", "multi_kernel", 4)
    builder.add_covariance("G_diag", "diagonal", 1)
    builder.add_random_effect("u_kron", ["cluster", "t1e1", "t1e2", "t2e1", "t2e2"], "G_kron")
    builder.add_random_effect("u_diag", ["cluster"], "G_diag")
    return builder.build()


def _build_ordinal_model() -> "semx.ModelIR":
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "ordinal")
    builder.add_variable("x", semx.VariableKind.Exogenous, "")
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")
    builder.add_covariance("G_ord", "diagonal", 1)
    builder.add_random_effect("u_cluster", ["cluster"], "G_ord")
    return builder.build()


def _kronecker_product(a, a_dim, b, b_dim):
    result = [0.0] * (a_dim * b_dim * a_dim * b_dim)
    for i in range(a_dim):
        for j in range(a_dim):
            for p in range(b_dim):
                for q in range(b_dim):
                    row = i * b_dim + p
                    col = j * b_dim + q
                    result[row * (a_dim * b_dim) + col] = a[i * a_dim + j] * b[p * b_dim + q]
    return result


def _kronecker_data():
    return {
        "y": [0, 1, 0, 1, 0, 1, 0, 1],
        "x": [-1.6, -1.1, -0.6, -0.1, 0.4, 0.9, 1.4, 1.9],
        "cluster": [1, 2, 1, 2, 1, 2, 1, 2],
        "t1e1": [1, 0, 0, 0, 1, 0, 0, 0],
        "t1e2": [0, 1, 0, 0, 0, 1, 0, 0],
        "t2e1": [0, 0, 1, 0, 0, 0, 1, 0],
        "t2e2": [0, 0, 0, 1, 0, 0, 0, 1],
    }


def _kronecker_fixed_covariance():
    trait_cov = [1.0, 0.35, 0.35, 1.0]
    env_cov = [1.0, 0.2, 0.2, 1.0]
    identity2 = [1.0, 0.0, 0.0, 1.0]
    return {
        "G_kron": [
            _kronecker_product(trait_cov, 2, identity2, 2),
            _kronecker_product(identity2, 2, env_cov, 2),
        ]
    }


def _negative_binomial_data():
    return {
        "y": [0, 1, 1, 2, 3, 2, 1, 4],
        "x": [-1.4, -0.9, -0.3, 0.2, 0.7, 1.1, 1.5, 1.9],
        "cluster": [1, 1, 2, 2, 3, 3, 4, 4],
    }


def _ordinal_data():
    return {
        "y": [0, 1, 1, 2, 1, 2, 2, 0],
        "x": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "cluster": [1, 1, 2, 2, 3, 3, 4, 4],
    }


def _ordinal_thresholds():
    return [-0.4, 0.6]


def test_laplace_gradient_matches_finite_difference():
    """Laplacian gradients exposed via evaluate_model_gradient should match FD checks."""
    model = _build_binomial_model()
    driver = semx.LikelihoodDriver()

    y = [0, 1, 0, 1, 0, 1]
    x = [-1.0, -0.5, 0.2, 0.7, 1.0, 1.5]
    cluster = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

    beta = 0.8
    sigma = 0.6

    data = {"y": y, "x": x, "cluster": cluster}
    linear_predictors = {"y": [beta * value for value in x]}
    dispersions = {"y": [1.0] * len(y)}
    covariance_parameters = {"G": [sigma]}

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
    )

    def loglik(beta_val: float, sigma_val: float) -> float:
        lp = {"y": [beta_val * value for value in x]}
        cov = {"G": [sigma_val]}
        return driver.evaluate_model_loglik(
            model,
            data,
            lp,
            dispersions,
            cov,
        )

    eps = 1e-5
    fd_beta = (loglik(beta + eps, sigma) - loglik(beta - eps, sigma)) / (2 * eps)
    fd_sigma = (loglik(beta, sigma + eps) - loglik(beta, sigma - eps)) / (2 * eps)

    assert gradients["beta"] == pytest.approx(fd_beta, rel=1e-4, abs=1e-6)
    assert gradients["G_0"] == pytest.approx(fd_sigma, rel=1e-4, abs=1e-6)


def test_negative_binomial_laplace_gradient_matches_finite_difference():
    model = _build_negative_binomial_model()
    driver = semx.LikelihoodDriver()

    data = _negative_binomial_data()

    options = semx.OptimizationOptions()
    options.max_iterations = 400
    options.tolerance = 5e-4
    options.learning_rate = 0.1

    result = driver.fit(model, data, options, "lbfgs")
    assert result.optimization_result.converged
    assert len(result.optimization_result.parameters) == 2
    beta, sigma = result.optimization_result.parameters

    dispersions = {"y": [1.0] * len(data["y"])}
    linear_predictors = {"y": [beta * value for value in data["x"]]}
    covariance_parameters = {"G_nb": [sigma]}

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
    )

    def loglik(beta_val: float, sigma_val: float) -> float:
        lp = {"y": [beta_val * value for value in data["x"]]}
        cov = {"G_nb": [sigma_val]}
        return driver.evaluate_model_loglik(
            model,
            data,
            lp,
            dispersions,
            cov,
        )

    step_beta = 5e-4
    step_sigma = max(1e-5, 0.1 * sigma)

    fd_beta = (loglik(beta + step_beta, sigma) - loglik(beta - step_beta, sigma)) / (2 * step_beta)
    fd_sigma = (loglik(beta, sigma + step_sigma) - loglik(beta, sigma - step_sigma)) / (2 * step_sigma)

    assert gradients["beta"] == pytest.approx(fd_beta, abs=2e-3)
    assert gradients["G_nb_0"] == pytest.approx(fd_sigma, abs=2e-3)


def test_ordinal_laplace_gradient_matches_finite_difference():
    model = _build_ordinal_model()
    driver = semx.LikelihoodDriver()

    data = _ordinal_data()
    thresholds = _ordinal_thresholds()

    beta = 0.7
    sigma = 0.8

    linear_predictors = {"y": [beta * value for value in data["x"]]}
    dispersions = {"y": [1.0] * len(data["y"])}
    covariance_parameters = {"G_ord": [sigma]}
    extra_params = {"y": thresholds}

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        extra_params=extra_params,
    )

    def loglik(beta_val: float, sigma_val: float) -> float:
        lp = {"y": [beta_val * value for value in data["x"]]}
        cov = {"G_ord": [sigma_val]}
        return driver.evaluate_model_loglik(
            model,
            data,
            lp,
            dispersions,
            cov,
            extra_params=extra_params,
        )

    step_beta = 5e-4
    step_sigma = max(1e-5, 0.1 * sigma)

    fd_beta = (loglik(beta + step_beta, sigma) - loglik(beta - step_beta, sigma)) / (2 * step_beta)
    fd_sigma = (loglik(beta, sigma + step_sigma) - loglik(beta, sigma - step_sigma)) / (2 * step_sigma)

    assert gradients["beta"] == pytest.approx(fd_beta, abs=2e-3)
    assert gradients["G_ord_0"] == pytest.approx(fd_sigma, abs=2e-3)


def test_kronecker_laplace_gradient_matches_finite_difference():
    model = _build_kronecker_model()
    driver = semx.LikelihoodDriver()

    data = _kronecker_data()
    fixed_covariance = _kronecker_fixed_covariance()

    options = semx.OptimizationOptions()
    options.max_iterations = 600
    options.tolerance = 5e-4
    options.learning_rate = 0.2

    result = driver.fit(model, data, options, "lbfgs", fixed_covariance_data=fixed_covariance)
    assert result.optimization_result.converged
    assert len(result.optimization_result.parameters) == 5

    beta, sigma_kron, weight_trait, weight_env, sigma_diag = result.optimization_result.parameters

    def loglik(beta_val, sigma_kron_val, weight_trait_val, weight_env_val, sigma_diag_val):
        linear_predictors = {"y": [beta_val * value for value in data["x"]]}
        covariance_parameters = {
            "G_kron": [sigma_kron_val, weight_trait_val, weight_env_val],
            "G_diag": [sigma_diag_val],
        }
        dispersions = {"y": [1.0] * len(data["y"])}
        return driver.evaluate_model_loglik(
            model,
            data,
            linear_predictors,
            dispersions,
            covariance_parameters,
            fixed_covariance_data=fixed_covariance,
        )

    dispersions = {"y": [1.0] * len(data["y"])}
    linear_predictors = {"y": [beta * value for value in data["x"]]}
    covariance_parameters = {
        "G_kron": [sigma_kron, weight_trait, weight_env],
        "G_diag": [sigma_diag],
    }

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        fixed_covariance_data=fixed_covariance,
    )

    step_fixed = 5e-4
    step_sigma = 1e-4

    fd_beta = (loglik(beta + step_fixed, sigma_kron, weight_trait, weight_env, sigma_diag) -
               loglik(beta - step_fixed, sigma_kron, weight_trait, weight_env, sigma_diag)) / (2 * step_fixed)
    fd_trait = (loglik(beta, sigma_kron, weight_trait + step_fixed, weight_env, sigma_diag) -
                loglik(beta, sigma_kron, weight_trait - step_fixed, weight_env, sigma_diag)) / (2 * step_fixed)
    fd_env = (loglik(beta, sigma_kron, weight_trait, weight_env + step_fixed, sigma_diag) -
              loglik(beta, sigma_kron, weight_trait, weight_env - step_fixed, sigma_diag)) / (2 * step_fixed)
    fd_sigma = (loglik(beta, sigma_kron + step_sigma, weight_trait, weight_env, sigma_diag) -
                loglik(beta, sigma_kron - step_sigma, weight_trait, weight_env, sigma_diag)) / (2 * step_sigma)
    fd_diag = (loglik(beta, sigma_kron, weight_trait, weight_env, sigma_diag + step_sigma) -
               loglik(beta, sigma_kron, weight_trait, weight_env, sigma_diag - step_sigma)) / (2 * step_sigma)

    assert gradients["beta"] == pytest.approx(fd_beta, abs=2e-3)
    assert gradients["G_kron_1"] == pytest.approx(fd_trait, abs=3e-3)
    assert gradients["G_kron_2"] == pytest.approx(fd_env, abs=3e-3)
    assert gradients["G_kron_0"] == pytest.approx(fd_sigma, abs=3e-3)
    assert gradients["G_diag_0"] == pytest.approx(fd_diag, abs=3e-3)
