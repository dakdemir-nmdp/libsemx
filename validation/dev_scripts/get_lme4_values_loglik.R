
if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4", repos = "https://cloud.r-project.org")
library(lme4)

data(sleepstudy, package = "lme4")
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML = FALSE)

print("Log Likelihood:")
print(logLik(fm1))
