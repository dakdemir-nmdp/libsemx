library(semx)
library(lme4)

data(sleepstudy)

# Define model using equations
equations <- c(
  "Reaction ~ Days + (Days | Subject)"
)

families <- c(Reaction = "gaussian")

print("Building model...")
model <- semx_model(
    equations = equations,
    families = families,
    data = sleepstudy
)

print("Fitting model...")
fit <- semx_fit(model, sleepstudy)

print("Extracting random effects...")
re <- semx_ranef(fit)
print(re)

# Check dimensions
if (length(re) != 1) stop("Expected 1 random effect block")
re_name <- names(re)[1]
print(paste("Checking random effect:", re_name))

if (nrow(re[[re_name]]) != length(unique(sleepstudy$Subject))) stop("Row count mismatch")
if (ncol(re[[re_name]]) != 2) stop("Column count mismatch")

print("Ranef check passed!")
