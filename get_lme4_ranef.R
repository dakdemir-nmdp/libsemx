library(lme4)
data(sleepstudy)

# Gaussian LMM
m1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML=FALSE)
re <- ranef(m1)$Subject
print("Gaussian Random Effects (Intercept, Days):")
print(head(re))
# Print all for verification if needed, or just the first few.
# We want to check specific values.
# Subject 308:
print(paste("Subject 308 Intercept:", re["308", "(Intercept)"]))
print(paste("Subject 308 Days:", re["308", "Days"]))

# Binomial GLMM
sleepstudy$ReactionBin <- ifelse(sleepstudy$Reaction > 250, 1, 0)
m2 <- glmer(ReactionBin ~ Days + (1 | Subject), sleepstudy, family=binomial)
re2 <- ranef(m2)$Subject
print("Binomial Random Effects (Intercept):")
print(head(re2))
print(paste("Subject 308 Intercept:", re2["308", "(Intercept)"]))
