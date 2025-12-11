
if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4", repos = "https://cloud.r-project.org")
library(lme4)

data(sleepstudy, package = "lme4")
# Fit the model: Reaction ~ Days + (Days | Subject)
# We use REML = FALSE to compare with ML estimation which is likely what libsemx does by default or can be configured to do.
# The user mentioned "Use ML for comparison" in the Rmd.
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML = FALSE)

print("Fixed Effects:")
print(fixef(fm1))

print("Variance Components:")
print(VarCorr(fm1))

print("Sigma:")
print(sigma(fm1))
