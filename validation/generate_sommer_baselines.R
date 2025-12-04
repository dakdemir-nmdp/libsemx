if (!require("sommer")) install.packages("sommer", repos="https://cloud.r-project.org")
library(sommer)
library(MASS)

set.seed(123)
n_geno <- 50
n_rep <- 4 # Increased reps to ensure convergence
N <- n_geno * n_rep

# Genotypes
id <- paste0("G", 1:n_geno)
# Relationship matrix (random positive definite)
X_marker <- matrix(rnorm(n_geno * 50), n_geno, 50)
K <- tcrossprod(X_marker)
K <- K / mean(diag(K))
rownames(K) <- colnames(K) <- id

# Data frame
df <- data.frame(
  id = rep(id, each = n_rep),
  env = rep(c("E1", "E2"), times = N/2)
)

# Simulate phenotype for GBLUP
# Var(u) = 2, Var(e) = 1
u <- mvrnorm(1, mu = rep(0, n_geno), Sigma = 2 * K)
names(u) <- id
df$y <- 10 + u[df$id] + rnorm(N)

# 1. GBLUP
# y ~ 1 + (1|id) with K
ans1 <- mmer(y ~ 1,
             random = ~ vs(id, Gu=K),
             rcov = ~ units,
             data = df, verbose = FALSE, date.warning = FALSE)

print("--- GBLUP Results ---")
print(summary(ans1)$varcomp)
print(paste("AIC:", ans1$AIC))
print(paste("BIC:", ans1$BIC))
print(paste("LogLik:", as.numeric(ans1$logLik)))
print("Fixed Effects:")
print(ans1$Beta)
print("BLUPs (head):")
print(head(ans1$U[[1]]$y))

# 2. GxE (Multi-Environment)
# y ~ env + (env|id)
# Covariance structure: Unstructured for Env (2x2), K for Id (10x10) -> Kronecker
# Residual: Homogeneous
# Simulate GxE data
Sigma_env <- matrix(c(2.0, 0.5, 0.5, 3.0), 2, 2)
u_gxe <- mvrnorm(n_geno, mu = c(0, 0), Sigma = Sigma_env) # n_geno x 2
rownames(u_gxe) <- id
# Correlate with K? Ideally u_vec ~ N(0, Sigma_env (x) K)
# To simulate: u_vec = chol(Sigma_env (x) K) * z
Sigma_full <- kronecker(Sigma_env, K)
u_vec <- mvrnorm(1, mu = rep(0, 2*n_geno), Sigma = Sigma_full)
# Map back to dataframe
# u_vec is ordered as E1:G1, E1:G2... E2:G1... or G1:E1... depending on kronecker order.
# kronecker(Sigma_env, K) implies Env is outer, Id is inner?
# Let's assume Env outer (E1 block, E2 block).
u_e1 <- u_vec[1:n_geno]
u_e2 <- u_vec[(n_geno+1):(2*n_geno)]
names(u_e1) <- id
names(u_e2) <- id

df$y_gxe <- 10 + (df$env == "E2")*2 + 
            ifelse(df$env=="E1", u_e1[df$id], u_e2[df$id]) + 
            rnorm(N)

# Try vs(env, id, Gu=K) which might default to unstructured or diagonal
ans2 <- mmer(y_gxe ~ env,
             random = ~ vs(usr(env), id, Gu=K),
             rcov = ~ units,
             data = df, verbose = FALSE, date.warning = FALSE)

print("--- GxE Results ---")
print(summary(ans2)$varcomp)
print(paste("AIC:", ans2$AIC))
print(paste("BIC:", ans2$BIC))
# Calculate LogLik from AIC: AIC = 2k - 2LL => LL = k - AIC/2
# But we need k.
# summary(ans2)$varcomp has rows. Fixed effects?
# Let's just use AIC/BIC for validation.
print("Fixed Effects:")
print(ans2$Beta)
print("BLUPs (head):")
# sommer returns BLUPs
print(head(ans2$U[[1]]$y_gxe))

# Save data
write.csv(df, "../data/sommer_gxe.csv", row.names = FALSE)
write.csv(as.data.frame(K), "../data/sommer_K.csv", row.names = TRUE)
