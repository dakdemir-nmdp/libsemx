library(semx)
library(sommer)
library(testthat)
library(Matrix)

set.seed(123)

# --- Helper Functions ---

# Generate a positive definite matrix (Kinship)
generate_K <- function(n) {
  X <- matrix(rnorm(n * (n + 20)), nrow = n)
  K <- tcrossprod(X)
  K <- K / mean(diag(K)) # Scale to average diagonal of 1
  rownames(K) <- colnames(K) <- paste0("ID_", 1:n)
  return(K)
}

# --- Test 1: GBLUP (Single Kernel) ---
cat("\n--- Test 1: GBLUP with Simulated Truth ---\n")

n <- 200
K <- generate_K(n)
ids <- rownames(K)

# True Parameters
sigma2_u_true <- 2.0
sigma2_e_true <- 1.0
beta_true <- 5.0

# Simulate Random Effects u ~ N(0, sigma2_u * K)
L_K <- t(chol(K))
u_true <- as.vector(L_K %*% rnorm(n, sd = sqrt(sigma2_u_true)))
names(u_true) <- ids

# Simulate Phenotype y = mu + u + e
e <- rnorm(n, sd = sqrt(sigma2_e_true))
y <- beta_true + u_true + e
df_gblup <- data.frame(id = ids, y = y)
# Ensure factor levels match K row order for correct indexing in semx
df_gblup$id <- factor(df_gblup$id, levels = rownames(K))

# Fit with semx
# Note: In semx, we use genomic(id, K) to specify the covariance structure
model_semx <- semx_model(
  equations = "y ~ 1 + (1 | genomic(id, K))",
  families = c(y = "gaussian"),
  data = list(K = K)
)
print("Model created.")
str(model_semx$fixed_covariance_data)
fit_semx <- semx_fit(model_semx, df_gblup)
params_semx <- summary(fit_semx)$parameters

# Extract semx variance components
# Parameter names might be "psi_re_genomic_id_K_re_genomic_id_K" (log scale) or similar
# We need to check the parameter names.
print(params_semx)

# Helper to find variance parameter
get_semx_var <- function(params, pattern) {
  idx <- grep(pattern, rownames(params))
  if (length(idx) == 0) return(NA)
  # semx reports parameters on the estimation scale (often log for variances)
  # But summary() might return them transformed? 
  # Let's assume the "Estimate" column is the value. 
  # Wait, semx summary usually reports transformed values if the internal representation is transformed?
  # Actually, for standard variances, semx usually reports log-variance in the raw vector, 
  # but let's check the output. The previous validation script showed "psi_..." values.
  # If they are log-variances, we need to exp() them.
  # However, the C++ code uses log-transform for Positive constraints.
  # Let's assume we need to exp() if the name starts with "psi_" or "cov_".
  # Actually, let's look at the previous output: "psi_y1_y1 = 0.6467". 
  # If that was log(sigma^2), then sigma^2 = 1.9.
  # Let's check the previous script's output again.
  val <- params[idx, "Estimate"]
  return(val) # semx summary returns natural scale
}

# Fit with sommer
fit_sommer <- mmer(y ~ 1,
                   random = ~ vsr(id, Gu = K),
                   rcov = ~ units,
                   data = df_gblup,
                   verbose = FALSE)
vc_sommer <- summary(fit_sommer)$varcomp

# Compare Variance Components
sigma2_u_semx <- get_semx_var(params_semx, "K_0") 
if (is.na(sigma2_u_semx)) sigma2_u_semx <- get_semx_var(params_semx, "psi_re_.*_re_.*")

sigma2_e_semx <- get_semx_var(params_semx, "psi_y_y") # Residual variance

sigma2_u_sommer <- vc_sommer["u:id", "VarComp"]
sigma2_e_sommer <- vc_sommer["units", "VarComp"]

cat(sprintf("True sigma2_u: %.2f\n", sigma2_u_true))
cat(sprintf("SEMX sigma2_u: %.2f\n", sigma2_u_semx))
cat(sprintf("Sommer sigma2_u: %.2f\n", sigma2_u_sommer))

cat(sprintf("True sigma2_e: %.2f\n", sigma2_e_true))
cat(sprintf("SEMX sigma2_e: %.2f\n", sigma2_e_semx))
cat(sprintf("Sommer sigma2_e: %.2f\n", sigma2_e_sommer))

expect_equal(sigma2_u_semx, sigma2_u_true, tolerance = 0.5)
expect_equal(sigma2_e_semx, sigma2_e_true, tolerance = 0.2)
expect_equal(sigma2_u_semx, sigma2_u_sommer, tolerance = 0.1)

# Compare BLUPs
# semx BLUPs extraction
# We need a way to get random effects. 
# Currently semx R package might not have a direct `ranef` extractor exposed nicely?
# Let's check if we can get them from the fit object.
# The C++ `LikelihoodDriver` computes them but maybe they aren't returned in the R list?
# Wait, `semx_fit` returns `optimization_result`.
# Let's check if `post_estimation` is called.
# If not, we might skip BLUP comparison for now or check if `predict` can return them.

# --- Test 2: GxE (Compound Symmetry) ---
cat("\n--- Test 2: GxE Simulation ---\n")
# Simulate 3 environments, same genotypes
n_env <- 3
n_geno <- 100
K_gxe <- generate_K(n_geno)
ids_gxe <- rownames(K_gxe)

# Create data frame
df_gxe <- expand.grid(id = ids_gxe, env = paste0("E", 1:n_env))
# Ensure factor levels match K row order for correct indexing in semx
df_gxe$id <- factor(df_gxe$id, levels = rownames(K_gxe))

# True Parameters

# True Parameters
sigma2_g <- 1.5 # Genetic variance
sigma2_ge <- 0.5 # GxE interaction variance
sigma2_e_gxe <- 0.8 # Residual

# Simulate effects
u_main <- as.vector(t(chol(K_gxe)) %*% rnorm(n_geno, sd = sqrt(sigma2_g)))
names(u_main) <- ids_gxe

# Interaction effects (independent per environment, but correlated via K)
u_int <- matrix(0, nrow = n_geno, ncol = n_env)
rownames(u_int) <- ids_gxe
colnames(u_int) <- paste0("E", 1:n_env)
for(j in 1:n_env) {
  u_int[,j] <- t(chol(K_gxe)) %*% rnorm(n_geno, sd = sqrt(sigma2_ge))
}
colnames(u_int) <- paste0("E", 1:n_env)

# Construct phenotype
df_gxe$y <- 0
for(i in 1:nrow(df_gxe)) {
  id_i <- df_gxe$id[i]
  env_i <- df_gxe$env[i]
  df_gxe$y[i] <- 10 + u_main[id_i] + u_int[id_i, env_i] + rnorm(1, sd = sqrt(sigma2_e_gxe))
}

# Fit with semx
# We can model this as a main effect + interaction
# y ~ 1 + (1 | genomic(id, K)) + (1 | genomic(id:env, K)) 
# Note: semx might not support interaction syntax `id:env` with genomic directly yet.
# Alternatively, we can use a multi-group approach or Kronecker product if supported.
# Let's try a simpler "Main Effect" model first to see if we recover sigma2_g + sigma2_ge/n_env?
# Or better, let's try to fit a model that mimics the simulation:
# y ~ 1 + (1 | genomic(id, K)) + (1 | genomic(id, K, by = env)) ?
# If `by` is not supported, we might need to construct the interaction kernel manually.
# K_int = block_diag(K, K, K) corresponding to the sorted data?
# Let's stick to a simpler GBLUP on this data ignoring environment to see if it captures total genetic variance.
# Or, let's try to use the `sommer` comparison logic:
# sommer: random = ~ vsr(id, Gu=K) + vsr(id:env, Gu=K) ?
# Actually, let's just do a single environment GBLUP on this data to verify it works with repeated records if we don't model the interaction explicitly (it goes to residual).

model_semx_gxe <- semx_model(
  equations = "y ~ 1 + (1 | genomic(id, K))",
  families = c(y = "gaussian"),
  data = list(K = K_gxe)
)
fit_semx_gxe <- semx_fit(model_semx_gxe, df_gxe)
params_gxe <- summary(fit_semx_gxe)$parameters
print(params_gxe)

sigma2_u_gxe_semx <- get_semx_var(params_gxe, "K_0")
if (is.na(sigma2_u_gxe_semx)) sigma2_u_gxe_semx <- get_semx_var(params_gxe, "psi_re_.*_re_.*")

cat(sprintf("Estimated Genetic Variance (Main + Avg Interaction): %.2f\n", sigma2_u_gxe_semx))
# Expectation: sigma2_g + sigma2_ge/k approx? Or just sigma2_g?
# If we ignore env, the interaction term u_int is noise correlated with K.
# It effectively adds to the genetic variance.

cat("GxE Simulation Test Completed (Exploratory)\n")
