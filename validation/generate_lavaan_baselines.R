
# validation/generate_lavaan_baselines.R

# Install packages if not present (commented out to avoid accidental re-installs)
# install.packages("lavaan")
# install.packages("readr")

library(lavaan)

# Load data
bfi <- read.csv("data/bfi.csv")

# Select only the 25 items
items <- c(paste0("A", 1:5), paste0("C", 1:5), paste0("E", 1:5), paste0("N", 1:5), paste0("O", 1:5))
bfi_items <- bfi[, items]

# Remove rows with missing values (Listwise deletion)
bfi_complete <- na.omit(bfi_items)

# Save the complete dataset for C++ to use
write.csv(bfi_complete, "data/bfi_complete.csv", row.names = FALSE)

cat("Original N:", nrow(bfi), "\n")
cat("Complete N:", nrow(bfi_complete), "\n")

# Define the CFA model
# 5 factors, standard CFA structure
model <- '
  Agreeableness =~ A1 + A2 + A3 + A4 + A5
  Conscientiousness =~ C1 + C2 + C3 + C4 + C5
  Extraversion =~ E1 + E2 + E3 + E4 + E5
  Neuroticism =~ N1 + N2 + N3 + N4 + N5
  Openness =~ O1 + O2 + O3 + O4 + O5
'

# Fit the model
# std.lv = TRUE fixes latent variances to 1.0, which is often easier for comparison than fixing first loading to 1.0
# But libsemx might default to fixing first loading. Let's check libsemx defaults or capabilities.
# Usually fixing first loading is the default in lavaan.
# Let's stick to default lavaan behavior (marker variable method) where first loading is fixed to 1.
# Explicitly request meanstructure to match libsemx's explicit intercept modeling
fit <- cfa(model, data = bfi_complete, meanstructure = TRUE)

# Print summary
summary(fit, fit.measures = TRUE)

# Extract values for comparison
cat("\n--- COMPARISON VALUES ---\n")
cat("LogLik:", logLik(fit), "\n")
cat("AIC:", AIC(fit), "\n")
cat("BIC:", BIC(fit), "\n")
cat("CFI:", fitMeasures(fit, "cfi"), "\n")
cat("TLI:", fitMeasures(fit, "tli"), "\n")
cat("RMSEA:", fitMeasures(fit, "rmsea"), "\n")
cat("SRMR:", fitMeasures(fit, "srmr"), "\n")

# Parameter estimates
pe <- parameterEstimates(fit)
# Filter for loadings (op "=~"), intercepts (op "~1"), variances (op "~~")
# We might want to output these to a file or just print them to copy-paste into the C++ test.
# For automation, printing is fine for now.

cat("\n--- PARAMETER ESTIMATES ---\n")
print(pe[pe$op %in% c("=~", "~~", "~1"), c("lhs", "op", "rhs", "est")])
