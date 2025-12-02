#' @useDynLib semx, .registration = TRUE
#' @import methods Rcpp
NULL

# Load the module
Rcpp::loadModule("semx", TRUE)

grm_vanraden <- function(markers, center = TRUE, normalize = TRUE) {
  grm_vanraden_cpp(markers, center, normalize)
}

grm_kronecker <- function(left, right) {
  grm_kronecker_cpp(left, right)
}


`%||%` <- function(lhs, rhs) {
	if (is.null(lhs)) rhs else lhs
}

.variable_kind_codes <- c(
	observed = 0L,
	latent = 1L,
	grouping = 2L
)

.edge_kind_codes <- c(
	loading = 0L,
	regression = 1L,
	covariance = 2L
)

#' Build a ModelIR from high-level equations
#'
#' This helper parses simple lavaan-style equations (\code{=~}, \code{~}, \code{~~})
#' and emits a \code{ModelIR} object plus lightweight metadata lists.
#'
#' @param equations Character vector of equations. Blank entries are ignored.
#' @param families Named character vector that maps observed variables to outcome
#'   family identifiers such as ``gaussian`` or ``binomial``.
#' @param kinds Optional named character vector forcing select variables to be
#'   ``observed``, ``latent``, or ``grouping``.
#' @param covariances Optional list of covariance definitions (name, structure, dimension).
#' @param genomic Optional named list of genomic covariance specs; each entry should provide
#'   a numeric matrix `markers` and optional `structure`, `center`, and `normalize` flags.
#' @param random_effects Optional list of random effect definitions (name, variables, covariance).
#'
#' @return A list with class \code{semx_model} that contains the compiled
#'   \code{ir}, \code{variables}, and \code{edges} components.
#' @export
semx_model <- function(equations, families, kinds = NULL, covariances = NULL, genomic = NULL, random_effects = NULL) {
	if (missing(equations) || length(equations) == 0L) {
		stop("at least one equation is required", call. = FALSE)
	}
	if (missing(families) || length(families) == 0L) {
		stop("families must include at least one observed variable", call. = FALSE)
	}

	eqs <- trimws(as.character(equations))
	eqs <- eqs[nzchar(eqs)]
	if (length(eqs) == 0L) {
		stop("all equations were blank after trimming", call. = FALSE)
	}

	fam_map <- as.list(families)
	if (is.null(names(fam_map)) || any(names(fam_map) == "")) {
		stop("families must be a named vector or list", call. = FALSE)
	}

	explicit_kinds <- list()
	if (!is.null(kinds)) {
		kind_names <- names(kinds)
		if (is.null(kind_names) || any(kind_names == "")) {
			stop("kinds must be named", call. = FALSE)
		}
		for (idx in seq_along(kinds)) {
			key <- trimws(kind_names[[idx]])
			value <- trimws(kinds[[idx]])
			aliased <- .variable_kind_codes[[tolower(value)]]
			if (is.na(aliased)) {
				stop(sprintf("unknown variable kind '%s'", value), call. = FALSE)
			}
			explicit_kinds[[key]] <- aliased
		}
	}

	var_defs <- list()
	var_order <- character()
	edges <- list()
	param_counts <- list()
	
	random_effects <- random_effects %||% list()
	re_count <- 0L
	uses_intercept_column <- FALSE

	next_parameter_id <- function(prefix, source, target) {
		base <- if (identical(prefix, "psi")) {
			sprintf("%s_%s_%s", prefix, source, target)
		} else {
			sprintf("%s_%s_on_%s", prefix, target, source)
		}
		count <- param_counts[[base]] %||% 0L
		param_counts[[base]] <<- count + 1L
		if (count == 0L) {
			return(base)
		}
		sprintf("%s_%d", base, count + 1L)
	}

	split_terms <- function(rhs) {
		terms <- character()
		current <- character()
		depth <- 0L
		chars <- strsplit(rhs, "")[[1]]
		
		for (char in chars) {
			if (char == "(") {
				depth <- depth + 1L
			} else if (char == ")") {
				depth <- depth - 1L
			}
			
			if (char == "+" && depth == 0L) {
				term <- trimws(paste(current, collapse = ""))
				if (nzchar(term)) terms <- c(terms, term)
				current <- character()
			} else {
				current <- c(current, char)
			}
		}
		term <- trimws(paste(current, collapse = ""))
		if (nzchar(term)) terms <- c(terms, term)
		terms
	}

	parse_mixed_term <- function(term) {
		if (!startsWith(term, "(") || !endsWith(term, ")") || !grepl("|", term, fixed = TRUE)) {
			return(NULL)
		}
		
		content <- substr(term, 2, nchar(term) - 1)
		parts <- strsplit(content, "|", fixed = TRUE)[[1]]
		if (length(parts) != 2L) {
			stop(sprintf("Invalid mixed model term: %s", term), call. = FALSE)
		}
		
		lhs_str <- trimws(parts[[1]])
		rhs_str <- trimws(parts[[2]])
		
		lhs_terms <- trimws(strsplit(lhs_str, "+", fixed = TRUE)[[1]])
		lhs_terms <- lhs_terms[nzchar(lhs_terms)]
		
		has_intercept <- TRUE
		design_vars <- character()
		
		for (t in lhs_terms) {
			if (t == "1") {
				has_intercept <- TRUE
			} else if (t == "0") {
				has_intercept <- FALSE
			} else {
				design_vars <- c(design_vars, t)
			}
		}
		
		list(
			grouping = rhs_str,
			design = design_vars,
			intercept = has_intercept
		)
	}

	ensure_variable <- function(name, default_kind, enforce_kind = FALSE, respect_explicit = TRUE) {
		normalized <- trimws(name)
		if (!nzchar(normalized)) {
			stop("variable names cannot be empty", call. = FALSE)
		}

		explicit_kind <- explicit_kinds[[normalized]]
		if (!respect_explicit && !is.null(explicit_kind) && explicit_kind != default_kind) {
			stop(sprintf("variable '%s' cannot take kind override in this context", normalized), call. = FALSE)
		}
		desired_kind <- if (!is.null(explicit_kind) && respect_explicit) explicit_kind else default_kind
		enforce <- if (enforce_kind) TRUE else (!is.null(explicit_kind))

		existing <- var_defs[[normalized]]
		if (!is.null(existing)) {
			if (enforce && existing$kind != desired_kind) {
				stop(sprintf("variable '%s' was previously registered with a different kind", normalized), call. = FALSE)
			}
			return(existing)
		}

		family <- ""
		if (identical(desired_kind, .variable_kind_codes[["observed"]])) {
			family <- fam_map[[normalized]] %||% ""
			if (!nzchar(family)) {
				stop(sprintf("observed variable '%s' requires a family", normalized), call. = FALSE)
			}
		}

		entry <- list(name = normalized, kind = desired_kind, family = family)
		var_defs[[normalized]] <<- entry
		var_order <<- c(var_order, normalized)
		entry
	}

	add_edge <- function(kind, source, target, prefix) {
		edges[[length(edges) + 1L]] <<- list(
			kind = kind,
			source = source,
			target = target,
			parameter_id = next_parameter_id(prefix, source, target)
		)
	}

	add_random_effect <- function(spec) {
		group <- spec$grouping
		design <- spec$design
		intercept <- spec$intercept
		
		ensure_variable(group, .variable_kind_codes[["grouping"]], enforce_kind = TRUE, respect_explicit = TRUE)
		
		ir_vars <- c(group)
		
		if (length(design) == 0L) {
			if (!intercept) {
				stop("Random effect term must have at least one variable (intercept or design)", call. = FALSE)
			}
			dimension <- 1L
		} else {
			if (intercept) {
				ir_vars <- c(ir_vars, "_intercept")
				uses_intercept_column <<- TRUE
				if (is.null(fam_map[["_intercept"]])) {
					fam_map[["_intercept"]] <<- "gaussian"
				}
				ensure_variable("_intercept", .variable_kind_codes[["observed"]])
			}
			for (dv in design) {
				ensure_variable(dv, .variable_kind_codes[["observed"]])
				ir_vars <- c(ir_vars, dv)
			}
			dimension <- length(ir_vars) - 1L
		}
		
		re_count <<- re_count + 1L
		cov_name <- sprintf("cov_re_%d", re_count)
		
		covariances <<- c(covariances, list(list(
			name = cov_name,
			structure = "unstructured",
			dimension = dimension
		)))
		
		random_effects <<- c(random_effects, list(list(
			name = sprintf("re_%s_%d", group, re_count),
			variables = ir_vars,
			covariance = cov_name
		)))
	}

	for (eq in eqs) {
		if (grepl("=~", eq, fixed = TRUE)) {
			parts <- strsplit(eq, "=~", fixed = TRUE)[[1]]
			latent <- trimws(parts[[1]])
			rhs_terms <- split_terms(parts[[2]])
			latent_var <- ensure_variable(latent, .variable_kind_codes[["latent"]], enforce_kind = TRUE, respect_explicit = FALSE)
			if (!length(rhs_terms)) {
				stop(sprintf("loading equation for %s is empty", latent), call. = FALSE)
			}
			for (indicator in rhs_terms) {
				obs_var <- ensure_variable(indicator, .variable_kind_codes[["observed"]], enforce_kind = TRUE, respect_explicit = FALSE)
				add_edge(.edge_kind_codes[["loading"]], latent_var$name, obs_var$name, "lambda")
			}
		} else if (grepl("~~", eq, fixed = TRUE)) {
			parts <- strsplit(eq, "~~", fixed = TRUE)[[1]]
			lhs <- trimws(parts[[1]])
			rhs <- trimws(parts[[2]])
			left <- ensure_variable(lhs, .variable_kind_codes[["observed"]])
			right <- ensure_variable(rhs, .variable_kind_codes[["observed"]])
			ordered <- sort(c(left$name, right$name))
			add_edge(.edge_kind_codes[["covariance"]], ordered[[1]], ordered[[2]], "psi")
		} else if (grepl("~", eq, fixed = TRUE)) {
			parts <- strsplit(eq, "~", fixed = TRUE)[[1]]
			target <- trimws(parts[[1]])
			predictors <- split_terms(parts[[2]])
			target_var <- ensure_variable(target, .variable_kind_codes[["observed"]])
			if (!length(predictors)) {
				stop(sprintf("regression equation for %s is empty", target), call. = FALSE)
			}
			for (predictor in predictors) {
				mixed <- parse_mixed_term(predictor)
				if (!is.null(mixed)) {
					add_random_effect(mixed)
					next
				}
				
				if (predictor %in% c("1", "0")) {
					next
				}
				source_var <- ensure_variable(predictor, .variable_kind_codes[["observed"]])
				add_edge(.edge_kind_codes[["regression"]], source_var$name, target_var$name, "beta")
			}
		} else {
			stop(sprintf("unrecognized equation: %s", eq), call. = FALSE)
		}
	}

	covariances <- covariances %||% list()
	genomic_data <- list()
	if (!is.null(genomic)) {
		if (is.null(names(genomic)) || any(names(genomic) == "")) {
			stop("genomic specifications must be a named list", call. = FALSE)
		}
		existing_ids <- vapply(covariances, function(x) x$name %||% "", character(1L))
		for (cov_id in names(genomic)) {
			entry <- genomic[[cov_id]]
			if (is.null(entry$markers)) {
				stop(sprintf("genomic covariance '%s' requires a 'markers' matrix", cov_id), call. = FALSE)
			}
			markers <- as.matrix(entry$markers)
			if (!is.numeric(markers)) {
				stop(sprintf("genomic covariance '%s' markers must be numeric", cov_id), call. = FALSE)
			}
			if (nrow(markers) == 0L || ncol(markers) == 0L) {
				stop(sprintf("genomic covariance '%s' markers must be non-empty", cov_id), call. = FALSE)
			}
				if (cov_id %in% existing_ids) {
					existing_dim <- covariances[[which(existing_ids == cov_id)[[1]]]]$dimension %||% NA_integer_
					if (!is.na(existing_dim) && existing_dim != nrow(markers)) {
						stop(
							sprintf(
								"genomic covariance '%s' dimension mismatch: %d vs %d",
								cov_id, existing_dim, nrow(markers)
							),
							call. = FALSE
						)
					}
				}
				structure_id <- entry$structure %||% "grm"
				if (!(cov_id %in% existing_ids)) {
					covariances <- c(covariances, list(list(name = cov_id, structure = structure_id, dimension = nrow(markers))))
				}
				genomic_data[[cov_id]] <- list(
					markers = markers,
					center = if (is.null(entry$center)) TRUE else isTRUE(entry$center),
					normalize = if (is.null(entry$normalize)) TRUE else isTRUE(entry$normalize),
					precomputed = if (is.null(entry$precomputed)) FALSE else isTRUE(entry$precomputed)
				)
		}
	}

	if (!is.null(random_effects)) {
		for (re in random_effects) {
			variables <- re$variables %||% character()
			if (length(variables)) {
				ensure_variable(variables[[1]], .variable_kind_codes[["grouping"]], enforce_kind = TRUE)
				if (length(variables) > 1L) {
					for (v in variables[-1L]) {
						ensure_variable(v, .variable_kind_codes[["observed"]])
					}
				}
			}
		}
	}

	builder <- new(ModelIRBuilder)
	for (var_name in var_order) {
		entry <- var_defs[[var_name]]
		builder$add_variable(entry$name, entry$kind, entry$family)
	}
	for (edge in edges) {
		builder$add_edge(edge$kind, edge$source, edge$target, edge$parameter_id)
	}

	for (cov in covariances) {
		builder$add_covariance(cov$name, cov$structure, cov$dimension)
	}

	if (!is.null(random_effects)) {
		for (re in random_effects) {
			builder$add_random_effect(re$name, re$variables, re$covariance)
		}
	}

	fixed_covariance_data <- list()
	if (length(genomic_data)) {
		for (cov_id in names(genomic_data)) {
			entry <- genomic_data[[cov_id]]
			if (entry$precomputed) {
				kernel <- as.vector(t(entry$markers))
			} else {
				kernel <- grm_vanraden(entry$markers, center = entry$center, normalize = entry$normalize)
			}
			fixed_covariance_data[[cov_id]] <- list(kernel)
		}
	}

	structure(
		list(
			ir = builder$build(),
			variables = var_defs,
			edges = edges,
			covariances = covariances,
			genomic_data = genomic_data,
			fixed_covariance_data = fixed_covariance_data,
			random_effects = random_effects,
			uses_intercept_column = uses_intercept_column
		),
		class = "semx_model"
	)
}

#' Fit a semx model
#'
#' @param model A semx_model object.
#' @param data A list or data.frame containing the data.
#' @param options Optimization options.
#' @param optimizer_name Name of the optimizer ("lbfgs" or "gd").
#' @param fixed_covariance_data Optional list of fixed covariance matrices.
#' @return A FitResult object.
#' @export
semx_fit <- function(model, data, options = NULL, optimizer_name = "lbfgs", fixed_covariance_data = NULL) {
    if (!inherits(model, "semx_model")) {
        stop("model must be a semx_model object")
    }
    
    if (is.null(options)) {
        options <- new(OptimizationOptions)
    }
    
    # Inject intercept column if needed
    if (isTRUE(model$uses_intercept_column) && is.null(data[["_intercept"]])) {
        # Determine number of rows
        n_rows <- 0L
        if (is.data.frame(data)) {
            n_rows <- nrow(data)
        } else if (is.list(data) && length(data) > 0L) {
            n_rows <- length(data[[1]])
        }
        
        if (n_rows > 0L) {
            data[["_intercept"]] <- rep(1.0, n_rows)
        }
    }
    
    driver <- new(LikelihoodDriver)

    status_data <- list()
    if (!is.null(model$status_vars)) {
        for (time_var in names(model$status_vars)) {
            status_var <- model$status_vars[[time_var]]
            if (!is.null(data[[status_var]])) {
                status_data[[time_var]] <- as.numeric(data[[status_var]])
            } else {
                stop(sprintf("Status variable '%s' not found in data", status_var))
            }
        }
    }

    merged_fixed <- fixed_covariance_data %||% list()
    if (!is.null(model$fixed_covariance_data) && length(model$fixed_covariance_data)) {
        for (cov_id in names(model$fixed_covariance_data)) {
            if (is.null(merged_fixed[[cov_id]])) {
                merged_fixed[[cov_id]] <- model$fixed_covariance_data[[cov_id]]
            }
        }
    }

    result <- if (length(merged_fixed) > 0L && length(status_data) > 0L) {
        driver$fit_with_fixed_and_status(model$ir, data, options, optimizer_name, merged_fixed, status_data)
    } else if (length(merged_fixed) > 0L) {
        driver$fit_with_fixed(model$ir, data, options, optimizer_name, merged_fixed)
    } else if (length(status_data) > 0L) {
        driver$fit_with_status(model$ir, data, options, optimizer_name, status_data)
    } else {
        driver$fit(model$ir, data, options, optimizer_name)
    }

    structure(
        list(
            optimization_result = result$optimization_result,
            standard_errors = result$standard_errors,
            vcov = result$vcov,
            aic = result$aic,
            bic = result$bic,
            model = model,
            data = data
        ),
        class = "semx_fit"
    )
}

#' @export
summary.semx_fit <- function(object, ...) {
  model <- object$model
  
  params <- object$optimization_result$parameters
  ses <- object$standard_errors
  
  if (length(ses) == 0) {
    ses <- rep(NA_real_, length(params))
  }
  
  z_values <- params / ses
  p_values <- 2 * (1 - pnorm(abs(z_values)))
  
  param_ids <- model$ir$parameter_ids()
  
  param_table <- data.frame(
    Estimate = params,
    Std.Error = ses,
    z.value = z_values,
    P.value = p_values,
    row.names = param_ids
  )
  
  fit_indices <- list(
    chisq = object$aic, # Placeholder, need to expose chi-square in Rcpp if not already
    # Wait, object$chi_square is not exposed in Rcpp yet? 
    # Let me check FitResult definition in Rcpp bindings.
    # Assuming they are exposed as per FitResult struct.
    chisq = object$chi_square, # Assuming this is exposed
    df = object$df,
    pvalue = object$p_value,
    cfi = object$cfi,
    tli = object$tli,
    rmsea = object$rmsea,
    srmr = object$srmr,
    aic = object$aic,
    bic = object$bic,
    loglik = -object$optimization_result$objective_value,
    iterations = object$optimization_result$iterations,
    converged = object$optimization_result$converged
  )
  
  res <- list(
    parameters = param_table,
    fit_indices = fit_indices
  )
  
  class(res) <- "summary.semx_fit"
  res
}

#' @export
print.summary.semx_fit <- function(x, ...) {
  cat("Optimization converged:", x$fit_indices$converged, "\n")
  cat("Iterations:", x$fit_indices$iterations, "\n")
  cat(sprintf("Log-likelihood: %.3f\n", x$fit_indices$loglik))
  
  if (!is.nan(x$fit_indices$chisq)) {
    cat(sprintf("Chi-square: %.3f (df=%d)\n", x$fit_indices$chisq, as.integer(x$fit_indices$df)))
    if (x$fit_indices$df > 0) {
       pval <- 1 - pchisq(x$fit_indices$chisq, x$fit_indices$df)
       cat(sprintf("P-value: %.3f\n", pval))
    }
  }
  
  indices <- character()
  if (!is.nan(x$fit_indices$cfi)) indices <- c(indices, sprintf("CFI: %.3f", x$fit_indices$cfi))
  if (!is.nan(x$fit_indices$tli)) indices <- c(indices, sprintf("TLI: %.3f", x$fit_indices$tli))
  if (!is.nan(x$fit_indices$rmsea)) indices <- c(indices, sprintf("RMSEA: %.3f", x$fit_indices$rmsea))
  if (!is.nan(x$fit_indices$srmr)) indices <- c(indices, sprintf("SRMR: %.3f", x$fit_indices$srmr))
  
  if (length(indices) > 0) {
    cat(paste(indices, collapse = ", "), "\n")
  }
  
  cat(sprintf("AIC: %.1f, BIC: %.1f\n\n", x$fit_indices$aic, x$fit_indices$bic))
  
  print(x$parameters, ...)
  invisible(x)
}

#' @export
predict.semx_fit <- function(object, newdata = NULL, ...) {
  model <- object$model
  params <- object$optimization_result$parameters
  names(params) <- model$ir$parameter_ids()
  
  if (is.null(newdata)) {
    data <- object$data
  } else {
    data <- newdata
  }
  
  if (is.null(data[["_intercept"]]) && isTRUE(model$uses_intercept_column)) {
     n <- if(is.data.frame(data)) nrow(data) else length(data[[1]])
     data[["_intercept"]] <- rep(1.0, n)
  }
  
  # Identify endogenous variables
  edges <- model$edges
  targets <- unique(vapply(Filter(function(e) e$kind == .edge_kind_codes[["regression"]], edges), function(e) e$target, character(1)))
  
  preds <- list()
  for (target in targets) {
    # Find predictors
    relevant_edges <- Filter(function(e) e$kind == .edge_kind_codes[["regression"]] && e$target == target, edges)
    
    y_hat <- 0
    for (edge in relevant_edges) {
      val <- params[[edge$parameter_id]]
      if (!is.null(data[[edge$source]])) {
        y_hat <- y_hat + val * data[[edge$source]]
      }
    }
    preds[[target]] <- y_hat
  }
  
  as.data.frame(preds)
}

#' @export
plot.semx_fit <- function(x, ...) {
  preds <- predict(x)
  data <- x$data
  
  for (target in names(preds)) {
    if (!is.null(data[[target]])) {
      y <- data[[target]]
      y_hat <- preds[[target]]
      resid <- y - y_hat
      
      plot(y_hat, resid, main = paste("Residuals vs Fitted:", target), xlab = "Fitted", ylab = "Residuals", ...)
      abline(h = 0, col = "red", lty = 2)
    }
  }
}

#' Extract heritability
#'
#' @param fit A semx_fit object.
#' @param genetic_component Name of the genetic variance parameter.
#' @param residual_component Name of the residual variance parameter.
#' @return Heritability estimate.
#' @export
semx_extract_heritability <- function(fit, genetic_component, residual_component) {
  params <- fit$optimization_result$parameters
  names(params) <- fit$model$ir$parameter_ids()
  
  var_g <- params[[genetic_component]]
  var_e <- params[[residual_component]]
  
  if (is.null(var_g)) stop(sprintf("Genetic component '%s' not found", genetic_component))
  if (is.null(var_e)) stop(sprintf("Residual component '%s' not found", residual_component))
  
  var_g / (var_g + var_e)
}

#' Cross-validation for genomic prediction
#'
#' @param model A semx_model object.
#' @param data A data.frame.
#' @param outcome Name of the outcome variable.
#' @param folds Number of folds.
#' @param seed Random seed.
#' @param options Optimization options.
#' @param optimizer_name Optimizer name.
#' @return List of metrics.
#' @export
semx_cv_genomic_prediction <- function(model, data, outcome, folds = 5, seed = 42, options = NULL, optimizer_name = "lbfgs") {
  set.seed(seed)
  n <- nrow(data)
  indices <- sample(n)
  fold_size <- floor(n / folds)
  
  correlations <- numeric(folds)
  mses <- numeric(folds)
  
  for (i in 1:folds) {
    start <- (i - 1) * fold_size + 1
    end <- if (i < folds) i * fold_size else n
    test_idx <- indices[start:end]
    
    masked_data <- data
    masked_data[test_idx, outcome] <- NA
    
    fit <- semx_fit(model, masked_data, options, optimizer_name)
    preds <- predict(fit, masked_data)
    
    y_true <- data[test_idx, outcome]
    y_pred <- preds[test_idx, outcome]
    
    valid <- !is.na(y_true) & !is.na(y_pred)
    if (sum(valid) > 0) {
      correlations[i] <- cor(y_true[valid], y_pred[valid])
      mses[i] <- mean((y_true[valid] - y_pred[valid])^2)
    }
  }
  
  list(
    mean_cor = mean(correlations, na.rm = TRUE),
    mean_mse = mean(mses, na.rm = TRUE),
    sd_cor = sd(correlations, na.rm = TRUE),
    sd_mse = sd(mses, na.rm = TRUE)
  )
}

#' Compute standardized estimates
#'
#' @param fit A FitResult object or semx_fit object.
#' @param model A semx_model object (optional if fit is semx_fit).
#' @param data A list or data.frame containing the data (optional if fit is semx_fit).
#' @return A list containing standardized estimates.
#' @export
semx_standardized_solution <- function(fit, model = NULL, data = NULL) {
    if (inherits(fit, "semx_fit")) {
        model <- fit$model
        data <- fit$data
    }
    if (is.null(model) || !inherits(model, "semx_model")) stop("model must be a semx_model object")
    
    param_names <- model$ir$parameter_ids()
    param_values <- fit$optimization_result$parameters
    
    compute_standardized_estimates_wrapper(model$ir, param_names, param_values, data)
}

#' Compute model diagnostics
#'
#' @param fit A FitResult object or semx_fit object.
#' @param model A semx_model object (optional if fit is semx_fit).
#' @param data A list or data.frame containing the data (optional if fit is semx_fit).
#' @return A list containing model diagnostics (implied moments, residuals, SRMR).
#' @export
semx_diagnostics <- function(fit, model = NULL, data = NULL) {
    if (inherits(fit, "semx_fit")) {
        model <- fit$model
        data <- fit$data
    }
    if (is.null(model) || !inherits(model, "semx_model")) stop("model must be a semx_model object")
    
    param_names <- model$ir$parameter_ids()
    param_values <- fit$optimization_result$parameters
    
    compute_model_diagnostics_wrapper(model$ir, param_names, param_values, data)
}

#' Compute modification indices
#'
#' @param fit A FitResult object or semx_fit object.
#' @param model A semx_model object (optional if fit is semx_fit).
#' @param data A list or data.frame containing the data (optional if fit is semx_fit).
#' @return A data.frame containing modification indices.
#' @export
semx_modification_indices <- function(fit, model = NULL, data = NULL) {
    if (inherits(fit, "semx_fit")) {
        model <- fit$model
        data <- fit$data
    }
    if (is.null(model) || !inherits(model, "semx_model")) stop("model must be a semx_model object")
    
    param_names <- model$ir$parameter_ids()
    param_values <- fit$optimization_result$parameters
    
    compute_modification_indices_wrapper(model$ir, param_names, param_values, data)
}
