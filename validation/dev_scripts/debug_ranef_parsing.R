library(semx)
library(lme4)

model_syntax <- "Reaction ~ Days + (1 | Subject)"
data(sleepstudy, package = "lme4")

print("Parsing model...")
model <- semx_model(model_syntax, data = sleepstudy)

print("Edges found:")
for (i in seq_along(model$edges)) {
    e <- model$edges[[i]]
    cat(sprintf("Edge %d: %s -> %s (Kind: %d)\n", i, e$source, e$target, e$kind))
}

print("Random Effects found:")
print(model$random_effects)
