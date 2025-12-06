
# Likelihood Comparison
ll_lav <- logLik(fit_lav)
ll_semx <- summ_semx$fit_indices$loglik
cat("\nLog-Likelihood Comparison:\n")
cat("Lavaan:", ll_lav, "\n")
cat("SEMX:  ", ll_semx, "\n")
cat("Diff:  ", as.numeric(ll_lav) - ll_semx, "\n")
