import math

import pytest
import semx


def _build_binomial_mixed_model():
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "binomial")
    builder.add_variable("x", semx.VariableKind.Latent)
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")
    builder.add_covariance("G", "diagonal", 1)
    builder.add_random_effect("u", ["cluster"], "G")
    return builder.build()


def _build_multi_effect_binomial_model():
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "binomial")
    builder.add_variable("x", semx.VariableKind.Latent)
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_variable("batch", semx.VariableKind.Grouping)
    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")
    builder.add_covariance("G_cluster", "diagonal", 1)
    builder.add_covariance("G_batch", "diagonal", 1)
    builder.add_random_effect("u_cluster", ["cluster"], "G_cluster")
    builder.add_random_effect("u_batch", ["batch"], "G_batch")
    return builder.build()


def _build_mixed_covariance_model():
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "binomial")
    builder.add_variable("x", semx.VariableKind.Latent)
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_variable("batch", semx.VariableKind.Grouping)
    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")
    builder.add_covariance("G_cluster", "diagonal", 1)
    builder.add_covariance("G_batch_fixed", "scaled_fixed", 1)
    builder.add_random_effect("u_cluster", ["cluster"], "G_cluster")
    builder.add_random_effect("u_batch", ["batch"], "G_batch_fixed")
    return builder.build()


def _build_random_slope_model():
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "binomial")
    builder.add_variable("x", semx.VariableKind.Latent)
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_variable("intercept_col", semx.VariableKind.Latent)
    builder.add_variable("z", semx.VariableKind.Latent)
    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")
    builder.add_covariance("G_cluster2", "unstructured", 2)
    builder.add_random_effect("u_cluster2", ["cluster", "intercept_col", "z"], "G_cluster2")
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


def _build_kronecker_model():
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "binomial")
    builder.add_variable("x", semx.VariableKind.Latent)
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_variable("t1e1", semx.VariableKind.Latent)
    builder.add_variable("t1e2", semx.VariableKind.Latent)
    builder.add_variable("t2e1", semx.VariableKind.Latent)
    builder.add_variable("t2e2", semx.VariableKind.Latent)
    builder.add_edge(semx.EdgeKind.Regression, "x", "y", "beta")
    builder.add_covariance("G_kron", "multi_kernel", 4)
    builder.add_covariance("G_diag", "diagonal", 1)
    builder.add_random_effect("u_kron", ["cluster", "t1e1", "t1e2", "t2e1", "t2e2"], "G_kron")
    builder.add_random_effect("u_diag", ["cluster"], "G_diag")
    return builder.build()

def test_fit_simple_regression():
    driver = semx.LikelihoodDriver()
    
    model = semx.ModelIR()
    model.variables = [
        semx.VariableSpec("y", semx.VariableKind.Observed, "gaussian")
    ]
    # x is not added as a variable because it's just a predictor
    
    model.edges = [
        semx.EdgeSpec(semx.EdgeKind.Regression, "x", "y", "beta")
    ]
    
    data = {
        "y": [1.0, 2.0, 3.0],
        "x": [1.0, 2.0, 3.0]
    }
    
    options = semx.OptimizationOptions()
    options.max_iterations = 100
    options.tolerance = 1e-6
    
    result = driver.fit(model, data, options, "lbfgs")
    
    assert result.converged
    assert len(result.parameters) == 1
    assert abs(result.parameters[0] - 1.0) < 1e-3


def test_gaussian_gradient_alignment_uses_parameter_specs():
    builder = semx.ModelIRBuilder()
    builder.add_variable("y", semx.VariableKind.Observed, "gaussian")
    builder.add_variable("intercept", semx.VariableKind.Observed, "gaussian")
    builder.add_variable("x1", semx.VariableKind.Observed, "gaussian")
    builder.add_variable("x2", semx.VariableKind.Observed, "gaussian")
    builder.add_variable("cluster", semx.VariableKind.Grouping)
    builder.add_edge(semx.EdgeKind.Regression, "intercept", "y", "beta_intercept")
    builder.add_edge(semx.EdgeKind.Regression, "x1", "y", "beta_x1")
    builder.add_edge(semx.EdgeKind.Regression, "x2", "y", "beta_x2")
    builder.add_covariance("G_cluster", "diagonal", 1)
    builder.add_random_effect("u_cluster", ["cluster"], "G_cluster")

    model = builder.build()
    driver = semx.LikelihoodDriver()

    data = {
        "y": [-0.4, 0.5, 0.2, 1.2, 1.8, 2.4],
        "intercept": [1.0] * 6,
        "x1": [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5],
        "x2": [0.3, -0.2, 0.4, -0.1, 0.7, -0.4],
        "cluster": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    }

    options = semx.OptimizationOptions()
    options.max_iterations = 400
    options.tolerance = 1e-6

    result = driver.fit(model, data, options, "lbfgs")
    assert result.converged

    param_ids = [spec.id for spec in model.parameters]
    assert param_ids == ["beta_intercept", "beta_x1", "beta_x2"]
    assert len(result.parameters) == len(param_ids) + 1

    beta_lookup = dict(zip(param_ids, result.parameters[: len(param_ids)]))
    sigma = result.parameters[len(param_ids)]

    def build_linear_predictors(beta_vals):
        return {
            "y": [
                beta_vals["beta_intercept"] * i
                + beta_vals["beta_x1"] * x1
                + beta_vals["beta_x2"] * x2
                for i, x1, x2 in zip(data["intercept"], data["x1"], data["x2"])
            ]
        }

    dispersions = {"y": [1.0] * len(data["y"])}

    def loglik(beta_vals, sigma_val):
        return driver.evaluate_model_loglik(
            model,
            data,
            build_linear_predictors(beta_vals),
            dispersions,
            {"G_cluster": [sigma_val]},
        )

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        build_linear_predictors(beta_lookup),
        dispersions,
        {"G_cluster": [sigma]},
    )

    beta_step = 5e-4
    for param_id in param_ids:
        beta_plus = dict(beta_lookup)
        beta_minus = dict(beta_lookup)
        beta_plus[param_id] += beta_step
        beta_minus[param_id] -= beta_step
        fd = (loglik(beta_plus, sigma) - loglik(beta_minus, sigma)) / (2 * beta_step)
        assert gradients[param_id] == pytest.approx(fd, abs=1e-3)

    theta_step = 1e-3
    sigma_plus = sigma * math.exp(theta_step)
    sigma_minus = sigma * math.exp(-theta_step)
    fd_sigma = (loglik(beta_lookup, sigma_plus) - loglik(beta_lookup, sigma_minus)) / (
        sigma_plus - sigma_minus
    )
    assert gradients["G_cluster_0"] == pytest.approx(fd_sigma, abs=5e-3)


def test_fit_binomial_mixed_model_laplace():
    model = _build_binomial_mixed_model()
    driver = semx.LikelihoodDriver()

    data = {
        "y": [0, 1, 0, 1, 0, 1],
        "x": [-1.0, -0.5, 0.2, 0.7, 1.0, 1.5],
        "cluster": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    }

    options = semx.OptimizationOptions()
    options.max_iterations = 200
    options.tolerance = 1e-4

    result = driver.fit(model, data, options, "lbfgs")
    assert result.converged
    assert len(result.parameters) == 2
    beta, sigma = result.parameters
    assert sigma > 0.0

    linear_predictors = {"y": [beta * val for val in data["x"]]}
    dispersions = {"y": [1.0] * len(data["y"])}
    covariance_parameters = {"G": [sigma]}

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
    )

    assert gradients["beta"] == pytest.approx(0.0, abs=1e-3)
    assert sigma < 1e-3
    assert gradients["G_0"] <= 0.0


def test_fit_multi_effect_binomial_laplace():
    model = _build_multi_effect_binomial_model()
    driver = semx.LikelihoodDriver()

    data = {
        "y": [0, 1, 0, 1, 0, 1, 0, 1],
        "x": [-1.5, -0.8, -0.2, 0.4, 0.9, 1.2, 1.5, 1.8],
        "cluster": [1, 1, 2, 2, 3, 3, 4, 4],
        "batch": [1, 1, 1, 1, 2, 2, 2, 2],
    }

    options = semx.OptimizationOptions()
    options.max_iterations = 250
    options.tolerance = 5e-4

    result = driver.fit(model, data, options, "lbfgs")
    assert result.converged
    assert len(result.parameters) == 3
    beta, sigma_cluster, sigma_batch = result.parameters
    assert sigma_cluster > 0.0
    assert sigma_batch > 0.0

    linear_predictors = {"y": [beta * val for val in data["x"]]}
    dispersions = {"y": [1.0] * len(data["y"])}
    covariance_parameters = {
        "G_cluster": [sigma_cluster],
        "G_batch": [sigma_batch],
    }

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
    )

    assert gradients["beta"] == pytest.approx(0.0, abs=1e-3)
    assert gradients["G_cluster_0"] <= 0.0
    assert gradients["G_batch_0"] <= 0.0


def test_fit_mixed_covariance_binomial_laplace():
    model = _build_mixed_covariance_model()
    driver = semx.LikelihoodDriver()

    data = {
        "y": [0, 1, 0, 1, 0, 1, 0, 1],
        "x": [-1.5, -0.8, -0.2, 0.4, 0.9, 1.2, 1.5, 1.8],
        "cluster": [1, 1, 2, 2, 3, 3, 4, 4],
        "batch": [1, 1, 1, 1, 2, 2, 2, 2],
    }

    fixed_covariance = {"G_batch_fixed": [[1.0]]}

    options = semx.OptimizationOptions()
    options.max_iterations = 250
    options.tolerance = 5e-4

    result = driver.fit(model, data, options, "lbfgs", fixed_covariance_data=fixed_covariance)
    assert result.converged
    assert len(result.parameters) == 3
    beta, sigma_cluster, sigma_batch_scale = result.parameters
    assert sigma_cluster > 0.0
    assert sigma_batch_scale > 0.0

    linear_predictors = {"y": [beta * val for val in data["x"]]}
    dispersions = {"y": [1.0] * len(data["y"])}
    covariance_parameters = {
        "G_cluster": [sigma_cluster],
        "G_batch_fixed": [sigma_batch_scale],
    }

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
        fixed_covariance_data=fixed_covariance,
    )

    assert gradients["beta"] == pytest.approx(0.0, abs=1e-3)
    assert gradients["G_cluster_0"] <= 0.0
    assert gradients["G_batch_fixed_0"] <= 0.0


def test_fit_random_slope_binomial_laplace():
    model = _build_random_slope_model()
    driver = semx.LikelihoodDriver()

    data = {
        "y": [0, 1, 0, 1, 0, 1, 0, 1],
        "x": [-1.4, -0.9, -0.3, 0.2, 0.7, 1.1, 1.5, 1.9],
        "cluster": [1, 1, 2, 2, 3, 3, 4, 4],
        "intercept_col": [1.0] * 8,
        "z": [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.3],
    }

    options = semx.OptimizationOptions()
    options.max_iterations = 300
    options.tolerance = 5e-4

    result = driver.fit(model, data, options, "lbfgs")
    assert result.converged
    assert len(result.parameters) == 4
    beta, sigma_intercept, cov_term, sigma_slope = result.parameters
    assert sigma_intercept > 0.0
    assert sigma_slope > 0.0
    assert abs(cov_term) < 1.0

    linear_predictors = {"y": [beta * val for val in data["x"]]}
    dispersions = {"y": [1.0] * len(data["y"])}
    covariance_parameters = {
        "G_cluster2": [sigma_intercept, cov_term, sigma_slope],
    }

    gradients = driver.evaluate_model_gradient(
        model,
        data,
        linear_predictors,
        dispersions,
        covariance_parameters,
    )

    assert gradients["beta"] == pytest.approx(0.0, abs=1e-3)
    assert gradients["G_cluster2_0"] <= 0.0
    assert gradients["G_cluster2_1"] == pytest.approx(0.0, abs=5e-3)
    assert gradients["G_cluster2_2"] <= 0.0


def test_fit_kronecker_binomial_laplace():
    model = _build_kronecker_model()
    driver = semx.LikelihoodDriver()

    data = {
        "y": [0, 1, 0, 1, 0, 1, 0, 1],
        "x": [-1.6, -1.1, -0.6, -0.1, 0.4, 0.9, 1.4, 1.9],
        "cluster": [1, 2, 1, 2, 1, 2, 1, 2],
        "t1e1": [1, 0, 0, 0, 1, 0, 0, 0],
        "t1e2": [0, 1, 0, 0, 0, 1, 0, 0],
        "t2e1": [0, 0, 1, 0, 0, 0, 1, 0],
        "t2e2": [0, 0, 0, 1, 0, 0, 0, 1],
    }

    trait_cov = [1.0, 0.35, 0.35, 1.0]
    env_cov = [1.0, 0.2, 0.2, 1.0]
    identity2 = [1.0, 0.0, 0.0, 1.0]

    fixed_covariance = {
        "G_kron": [
            _kronecker_product(trait_cov, 2, identity2, 2),
            _kronecker_product(identity2, 2, env_cov, 2),
        ]
    }

    options = semx.OptimizationOptions()
    options.max_iterations = 600
    options.tolerance = 5e-4
    options.learning_rate = 0.2

    result = driver.fit(model, data, options, "lbfgs", fixed_covariance_data=fixed_covariance)
    assert result.converged
    assert len(result.parameters) == 5
    beta, sigma_kron, weight_trait, weight_env, sigma_diag = result.parameters
    assert sigma_kron > 0.0
    assert sigma_diag > 0.0

    linear_predictors = {"y": [beta * val for val in data["x"]]}
    dispersions = {"y": [1.0] * len(data["y"])}
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

    assert gradients["beta"] == pytest.approx(0.0, abs=2e-3)
    assert gradients["G_kron_0"] <= 0.0
    assert gradients["G_kron_1"] == pytest.approx(0.0, abs=5e-3)
    assert gradients["G_kron_2"] == pytest.approx(0.0, abs=5e-3)
    assert gradients["G_diag_0"] <= 0.0
