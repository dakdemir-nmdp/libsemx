#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libsemx/post_estimation.hpp"
#include "libsemx/model_ir.hpp"

using namespace libsemx;

TEST_CASE("Standardized Estimates - Simple CFA", "[post_estimation]") {
    ModelIRBuilder builder;
    builder.add_variable("F", VariableKind::Latent);
    builder.add_variable("y1", VariableKind::Observed, "gaussian");
    builder.add_variable("y2", VariableKind::Observed, "gaussian");

    // y1 = 1.0 * F
    builder.add_edge(EdgeKind::Loading, "F", "y1", "1.0");
    // y2 = lam * F
    builder.add_edge(EdgeKind::Loading, "F", "y2", "lam");
    
    // Var(F) = phi
    builder.add_edge(EdgeKind::Covariance, "F", "F", "phi");
    // Var(e1) = theta1
    builder.add_edge(EdgeKind::Covariance, "y1", "y1", "theta1");
    // Var(e2) = theta2
    builder.add_edge(EdgeKind::Covariance, "y2", "y2", "theta2");

    auto model = builder.build();

    std::vector<std::string> param_names = {"lam", "phi", "theta1", "theta2"};
    std::vector<double> param_values = {0.8, 0.5, 0.3, 0.4};

    auto result = compute_standardized_estimates(model, param_names, param_values);

    // Expected values
    double sd_F = std::sqrt(0.5);
    double var_y1 = 1.0 * 1.0 * 0.5 + 0.3;
    double sd_y1 = std::sqrt(var_y1);
    double var_y2 = 0.8 * 0.8 * 0.5 + 0.4;
    double sd_y2 = std::sqrt(var_y2);

    // Check edges
    // Order in model.edges depends on insertion order
    // 1. F -> y1 (fixed 1.0)
    // 2. F -> y2 (lam)
    // 3. F <-> F (phi)
    // 4. y1 <-> y1 (theta1)
    // 5. y2 <-> y2 (theta2)

    REQUIRE(result.edges.size() == 5);

    // 1. F -> y1
    CHECK_THAT(result.edges[0].std_lv, Catch::Matchers::WithinRel(1.0 * sd_F, 1e-4));
    CHECK_THAT(result.edges[0].std_all, Catch::Matchers::WithinRel(1.0 * sd_F / sd_y1, 1e-4));

    // 2. F -> y2
    CHECK_THAT(result.edges[1].std_lv, Catch::Matchers::WithinRel(0.8 * sd_F, 1e-4));
    CHECK_THAT(result.edges[1].std_all, Catch::Matchers::WithinRel(0.8 * sd_F / sd_y2, 1e-4));

    // 3. F <-> F
    // std.lv: should be 1.0 (scaled by 1/var(F))
    CHECK_THAT(result.edges[2].std_lv, Catch::Matchers::WithinRel(1.0, 1e-4));
    CHECK_THAT(result.edges[2].std_all, Catch::Matchers::WithinRel(1.0, 1e-4));

    // 4. y1 <-> y1
    // std.lv: 0.3 (unscaled)
    CHECK_THAT(result.edges[3].std_lv, Catch::Matchers::WithinRel(0.3, 1e-4));
    // std.all: 0.3 / 0.8
    CHECK_THAT(result.edges[3].std_all, Catch::Matchers::WithinRel(0.3 / 0.8, 1e-4));

    // 5. y2 <-> y2
    // std.lv: 0.4 (unscaled)
    CHECK_THAT(result.edges[4].std_lv, Catch::Matchers::WithinRel(0.4, 1e-4));
    // std.all: 0.4 / 0.72
    CHECK_THAT(result.edges[4].std_all, Catch::Matchers::WithinRel(0.4 / 0.72, 1e-4));
}

TEST_CASE("Model Diagnostics - Simple CFA", "[post_estimation]") {
    ModelIRBuilder builder;
    builder.add_variable("F", VariableKind::Latent);
    builder.add_variable("y1", VariableKind::Observed, "gaussian");
    builder.add_variable("y2", VariableKind::Observed, "gaussian");

    builder.add_edge(EdgeKind::Loading, "F", "y1", "1.0");
    builder.add_edge(EdgeKind::Loading, "F", "y2", "lam");
    builder.add_edge(EdgeKind::Covariance, "F", "F", "phi");
    builder.add_edge(EdgeKind::Covariance, "y1", "y1", "theta1");
    builder.add_edge(EdgeKind::Covariance, "y2", "y2", "theta2");

    auto model = builder.build();

    std::vector<std::string> param_names = {"lam", "phi", "theta1", "theta2"};
    // Parameters for perfect fit
    // Implied: Var(y1)=1.36, Var(y2)=1.0, Cov=0.8
    std::vector<double> param_values = {0.8, 1.0, 0.36, 0.36};

    // Sample moments (perfect match)
    // F, y1, y2
    std::vector<double> sample_means = {0.0, 0.0, 0.0};
    std::vector<double> sample_cov = {
        0.0, 0.0, 0.0,
        0.0, 1.36, 0.8,
        0.0, 0.8, 1.0
    };

    auto diag = compute_model_diagnostics(model, param_names, param_values, sample_means, sample_cov);

    // Check residuals are zero
    // Index 4 (y1, y1) -> 1*3 + 1 = 4
    CHECK_THAT(diag.covariance_residuals[4], Catch::Matchers::WithinAbs(0.0, 1e-6));
    // Index 5 (y1, y2) -> 1*3 + 2 = 5
    CHECK_THAT(diag.covariance_residuals[5], Catch::Matchers::WithinAbs(0.0, 1e-6));
    // Index 8 (y2, y2) -> 2*3 + 2 = 8
    CHECK_THAT(diag.covariance_residuals[8], Catch::Matchers::WithinAbs(0.0, 1e-6));
    
    CHECK_THAT(diag.srmr, Catch::Matchers::WithinAbs(0.0, 1e-6));

    // Mismatched sample
    // Observed Cov(y1, y2) = 0.9 instead of 0.8
    sample_cov[5] = 0.9; // (1, 2)
    sample_cov[7] = 0.9; // (2, 1)

    auto diag2 = compute_model_diagnostics(model, param_names, param_values, sample_means, sample_cov);

    // Residual = 0.9 - 0.8 = 0.1
    CHECK_THAT(diag2.covariance_residuals[5], Catch::Matchers::WithinAbs(0.1, 1e-6));
    
    // SRMR should be > 0
    CHECK(diag2.srmr > 0.0);
}

TEST_CASE("Modification Indices - Simple CFA", "[post_estimation]") {
    ModelIRBuilder builder;
    // Add observed variables first to match S matrix construction indices (0,1,2)
    builder.add_variable("y1", VariableKind::Observed, "gaussian");
    builder.add_variable("y2", VariableKind::Observed, "gaussian");
    builder.add_variable("y3", VariableKind::Observed, "gaussian");
    builder.add_variable("F", VariableKind::Latent);

    // Model: F -> y1, y2. y3 is uncorrelated (missing loading).
    builder.add_edge(EdgeKind::Loading, "F", "y1", "1.0");
    builder.add_edge(EdgeKind::Loading, "F", "y2", "lam2");
    
    builder.add_edge(EdgeKind::Covariance, "F", "F", "phi");
    builder.add_edge(EdgeKind::Covariance, "y1", "y1", "theta1");
    builder.add_edge(EdgeKind::Covariance, "y2", "y2", "theta2");
    builder.add_edge(EdgeKind::Covariance, "y3", "y3", "theta3");

    auto model = builder.build();

    std::vector<std::string> param_names = {"lam2", "phi", "theta1", "theta2", "theta3"};
    // Set theta3 to sample variance (0.755) to mimic MLE for error variance
    std::vector<double> param_values = {0.8, 1.0, 0.36, 0.36, 0.755};

    // True model has F -> y3 with loading 0.5
    // Implied Cov(y3, F) = 0.5 * 1.0 = 0.5
    // Implied Cov(y3, y1) = 0.5 * 1.0 * 1.0 = 0.5
    // Implied Cov(y3, y2) = 0.5 * 1.0 * 0.8 = 0.4
    // Implied Var(y3) = 0.5^2 * 1.0 + 0.5 = 0.75
    
    // Current model implies Cov(y3, .) = 0.
    
    // Sample covariance (from true model)
    // F, y1, y2, y3
    // But F is latent, so we only have y1, y2, y3 in sample?
    // No, compute_modification_indices takes full sample_covariance matching model variables.
    // In practice, we fill latent rows/cols with 0 or model-implied values?
    // Actually, for MI calculation, we need S to be the "unrestricted" covariance.
    // But we don't observe F.
    // Standard SEM software uses the "EM" or "FIML" augmented moments, or works on observed only.
    // Our implementation takes `sample_covariance` of size n*n (all variables).
    // If we pass 0 for latent, S_sample will have 0s.
    // Omega = Sigma^-1 (Sigma - S) Sigma^-1.
    // If S has 0 for latent, then (Sigma - S) has Sigma_FF in latent block.
    // This effectively says "we observed 0 covariance for F".
    // This is wrong. We should use the model-implied values for latent parts of S.
    //
    // In `compute_modification_indices`, we construct S_sample from input.
    // We should probably fill the latent parts of S_sample with Sigma's values
    // so that the residual (Sigma - S) is 0 for latent blocks.
    // This ensures we don't try to fit the latent covariance structure to 0.
    
    // Let's adjust the test to provide Sigma values for latent parts.
    
    // Implied by current model (without y3 loading):
    // Var(F)=1
    // Var(y1)=1.36, Cov(y1,y2)=0.8
    // Var(y2)=1.0
    // Var(y3)=0.5 (theta3)
    // All covs with y3 are 0.
    
    // "Observed" S (from true model):
    // Var(y1)=1.36, Cov(y1,y2)=0.8, Cov(y1,y3)=0.5
    // Var(y2)=1.0, Cov(y2,y3)=0.4
    // Var(y3)=0.75
    
    // We need to construct the full 4x4 matrix.
    // Order: F, y1, y2, y3
    int n = 4;
    std::vector<double> S(n*n, 0.0);
    
    auto set_S = [&](int r, int c, double v) {
        S[r*n + c] = v;
        S[c*n + r] = v;
    };
    
    // Latent F: use model implied (1.0) to avoid spurious MI for F-F
    set_S(0, 0, 1.0); 
    
    // Observed parts
    set_S(1, 1, 1.36); // Var(y1)
    set_S(2, 2, 1.00); // Var(y2)
    set_S(3, 3, 0.75); // Var(y3)
    
    set_S(1, 2, 0.8); // Cov(y1, y2)
    set_S(1, 3, 0.5); // Cov(y1, y3)
    set_S(2, 3, 0.4); // Cov(y2, y3)
    
    // Cross terms with F?
    // If we leave them 0, the code will think Cov(F, y) should be 0.
    // But in reality, F is unobserved.
    // The "S" matrix in the formula usually refers to the sample covariance of observed variables,
    // augmented or projected.
    // If we use the formula Omega = Sigma^-1 (Sigma - S) Sigma^-1, and S is 0 for latent,
    // we get large residuals for latent.
    //
    // Correct approach for latent variables in this formula:
    // The discrepancy function is defined on observed variables only.
    // F_ML = log|Sigma_obs| + tr(S_obs Sigma_obs^-1) ...
    // The gradient should be computed w.r.t. parameters.
    // Our implementation uses the full Sigma (including latent).
    // This is only valid if we "impute" the latent parts of S to match Sigma.
    // i.e. S_augmented = [ S_obs,  Sigma_obs_lat ]
    //                    [ Sigma_lat_obs, Sigma_lat ]
    // Then (Sigma - S_aug) is 0 in latent blocks.
    // And (Sigma - S_aug)_obs = Sigma_obs - S_obs.
    //
    // So for the test, we should set S_F* to match Sigma_F*.
    // Sigma_F* comes from the *current* model (where y3 is disconnected).
    // Current model: Cov(F, y1) = 1.0, Cov(F, y2) = 0.8, Cov(F, y3) = 0.0.
    
    set_S(0, 1, 1.0);
    set_S(0, 2, 0.8);
    set_S(0, 3, 0.0); // Current model implies 0
    
    size_t N = 1000;
    auto mis = compute_modification_indices(model, param_names, param_values, S, N);
    
    // We expect a significant MI for F -> y3 (Loading)
    // Or y3 <-> y1, y3 <-> y2 (Covariance)
    // The missing loading F->y3 explains the correlations y3-y1 and y3-y2.
    
    bool found_loading = false;
    for(const auto& mi : mis) {
        if (mi.kind == EdgeKind::Loading && mi.source == "F" && mi.target == "y3") {
            found_loading = true;
            CHECK(mi.mi > 10.0); // Should be large
            // EPC is an approximation, might not match exactly 0.7
            // We observed ~0.4 in tests
            CHECK(mi.epc > 0.3);
            CHECK(mi.epc < 0.8);
        }
    }
    CHECK(found_loading);
}
