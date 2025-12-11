library(semx)

model <- semx_model(
  equations = c("y ~ 1", "y ~~ 0.001 * y"),
  families = c(y = "gaussian")
)

print("Edges:")
for (e in model$edges) {
  cat(sprintf("Kind: %d, Source: %s, Target: %s, Param: %s\n", e$kind, e$source, e$target, e$parameter_id))
}
