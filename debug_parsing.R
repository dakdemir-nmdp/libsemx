library(semx)

model_syntax <- "
  F =~ NA*y1 + y2 + y3
  y1 ~~ y1
  y3 ~~ y3
  y2 ~~ 1*y2  # Fix residual variance for ordinal
  F ~~ 1*F    # Fix factor variance for scale
  y2 ~ 0*1    # Fix intercept to 0
"

families <- c(
  y1 = "gaussian",
  y2 = "ordinal",
  y3 = "gaussian"
)

# Mock data frame to get names (not strictly needed if families is complete, but good for completeness)
df <- data.frame(y1=rnorm(10), y2=ordered(sample(1:3, 10, replace=T)), y3=rnorm(10))

model <- semx:::semx_model(model_syntax, families, data=df)

cat("--- Edges ---\n")
print(model)

cat("\n--- Parameters ---\n")
print(modelscalar-export undefined array-tied-special scalar-tied-special scalar-export array-tied-special scalar-export integer-special scalar-export undefined integer-special scalar-export scalar-export scalar-tied-export-special integer-special scalar-export array-tied-special scalar-special association-readonly-hide-hideval-special scalar-export undefined scalar-tied-special array scalar-export undefined scalar-export scalar-export integer-special scalar integer-readonly-special undefined integer-readonly-special scalar-special integer-special integer-readonly-special integer-readonly-special array integer-special undefined scalar-export undefined undefined array array-readonly-special undefined scalar-export scalar-special scalar-export undefined scalar-export scalar-readonly-special scalar scalar-export scalar-special scalar-special scalar-readonly-tied-special undefined scalar scalar scalar association-readonly-hide-hideval-special undefined association-hide-hideval-special scalar-special scalar-special undefined scalar-special integer-readonly-special scalar scalar-special array undefined scalar-export scalar-special integer-special integer-readonly-special integer-special scalar-export scalar scalar-export array-readonly-special scalar-special scalar-special scalar-special scalar-special scalar-export array-readonly-tied-special scalar-special scalar-export integer-special scalar-export scalar-export scalar undefined array-tied-special scalar-export undefined undefined undefined scalar-tied-special scalar-export-special scalar-export undefined integer-readonly-special scalar-tied-special scalar-export scalar-export scalar scalar-export-special scalar-export undefined array-tied-special scalar undefined scalar-export scalar-tied-special undefined association scalar-export undefined integer-readonly-special scalar-export integer-readonly-special scalar-export integer-special scalar scalar-special scalar-export scalar-export scalar undefined scalar-export scalar-special scalar undefined scalar scalar-export undefined array-tied-special scalar array array scalar scalar-export integer scalar-export scalar-export scalar scalar-tied-special undefined scalar-export integer-export-special scalar-export-special scalar-export scalar-export integer-special scalar scalar-export array-special scalar-export scalar undefined scalar scalar undefined scalar-export integer-readonly-special integer scalar scalar-special array-tied-special undefined array-tied-special scalar undefined array scalar-special undefined scalar-export undefined array-special integer-readonly-special scalar-export integer-special undefined undefined scalar-export undefined scalar-export scalar-special scalar-tied-special integer scalar scalar scalar-special integer-special undefined scalar-export integer-special)
