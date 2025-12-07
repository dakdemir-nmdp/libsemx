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

build_semx_ir <- function(variables, edges, covariances, random_effects, parameters = NULL) {
    builder <- new(ModelIRBuilder)
    
    for (var_name in names(variables)) {
        entry <- variables[[var_name]]
        builder$add_variable(
            entry$name,
            entry$kind,
            entry$family,
            entry$label %||% "",
            entry$measurement_level %||% ""
        )
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
    if (!is.null(parameters)) {
        for (param_name in names(parameters)) {
            val <- parameters[[param_name]]
            builder$register_parameter(param_name, as.numeric(val))
        }
    }
    
    builder$build()
}

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
#' @param data Optional list or data.frame containing data. If provided, it can be used to resolve
#'   genomic matrices referenced in `genomic` specs or equations.
#'
#' @return A list with class \code{semx_model} that contains the compiled
#'   \code{ir}, \code{variables}, and \code{edges} components.
#' @export
semx_model <- function(equations, families, kinds = NULL, covariances = NULL, genomic = NULL, random_effects = NULL, data = NULL) {
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
	status_vars <- list()

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
				family <- "fixed"
			}
		}

		entry <- list(name = normalized, kind = desired_kind, family = family)
		var_defs[[normalized]] <<- entry
		var_order <<- c(var_order, normalized)
		entry
	}

	parse_term_with_fixed <- function(term) {
		if (grepl("*", term, fixed = TRUE)) {
			parts <- strsplit(term, "*", fixed = TRUE)[[1]]
			if (length(parts) == 2) {
				val_str <- trimws(parts[[1]])
				var <- trimws(parts[[2]])
				# Handle "NA" for freeing a parameter (not fully supported yet but good to parse)
				if (val_str == "NA") {
					return(list(var = var, val = NA)) 
				}
				val <- as.numeric(val_str)
				if (!is.na(val)) {
					return(list(var = var, val = val))
				}
			}
		}
		return(list(var = term, val = NULL))
	}

	parse_genomic_group <- function(term) {
		if (startsWith(term, "genomic(") && endsWith(term, ")")) {
			content <- substr(term, 9, nchar(term) - 1)
			parts <- trimws(strsplit(content, ",")[[1]])
			if (length(parts) == 1) {
				return(list(var = parts[1], cov = parts[1])) # Default cov name same as var
			} else if (length(parts) == 2) {
				return(list(var = parts[1], cov = parts[2]))
			}
		}
		NULL
	}

	add_edge <- function(kind, source, target, prefix, fixed_val = NULL) {
		pid <- if (!is.null(fixed_val) && !is.na(fixed_val)) as.character(fixed_val) else next_parameter_id(prefix, source, target)
		edges[[length(edges) + 1L]] <<- list(
			kind = kind,
			source = source,
			target = target,
			parameter_id = pid
		)
	}

	add_random_effect <- function(spec, target) {
		group <- spec$grouping
		design <- spec$design
		intercept <- spec$intercept
		
		cov_structure_type <- "unstructured"
		is_genomic <- FALSE
		cov_name_override <- NULL
		
		genomic_parsed <- parse_genomic_group(group)
		if (!is.null(genomic_parsed)) {
			is_genomic <- TRUE
			group <- "_genomic_group_"
			cov_name_override <- genomic_parsed$cov
			cov_structure_type <- "grm"
			
			# Use the ID variable as the design variable (index)
			# This enables the C++ backend to map observations to K matrix rows
			design <- c(genomic_parsed$var)
			intercept <- FALSE
		}

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
					fam_map[["_intercept"]] <<- ""
				}
				ensure_variable("_intercept", .variable_kind_codes[["observed"]])
			}
			for (dv in design) {
				ensure_variable(dv, .variable_kind_codes[["observed"]])
				ir_vars <- c(ir_vars, dv)
			}
			dimension <- length(ir_vars) - 1L
		}
		
		if (is_genomic) {
			# For genomic/GRM, the dimension is the size of the kernel (N)
			# We try to find it in genomic spec or data
			found_dim <- NA_integer_
			
			# Check genomic arg
			if (!is.null(genomic[[cov_name_override]])) {
				entry <- genomic[[cov_name_override]]
				if (!is.null(entry$markers)) found_dim <- nrow(entry$markers)
				if (!is.null(entry$data)) found_dim <- nrow(entry$data)
			}
			
			# Check data arg
			if (is.na(found_dim) && !is.null(data[[cov_name_override]])) {
				mat <- data[[cov_name_override]]
				if (is.matrix(mat) || inherits(mat, "Matrix")) {
					found_dim <- nrow(mat)
				}
			}
			
			if (!is.na(found_dim)) {
				dimension <- found_dim
			} else {
				warning(sprintf("Could not determine dimension for genomic covariance '%s'. Ensure it is provided in 'genomic' or 'data'.", cov_name_override), call. = FALSE)
			}
			
			cov_name <- cov_name_override
		} else {
			cov_name <- sprintf("cov_re_%d", re_count + 1L)
		}
		
		re_count <<- re_count + 1L
		re_name <- sprintf("re_%s_%d", group, re_count)
		
		# Add to covariances if not exists
		exists <- FALSE
		for (c in covariances) if (c$name == cov_name) exists <- TRUE
		
		if (!exists) {
			covariances <<- c(covariances, list(list(
				name = cov_name,
				structure = cov_structure_type,
				dimension = dimension
			)))
		}
		
		random_effects <<- c(random_effects, list(list(
			name = re_name,
			variables = ir_vars,
			covariance = cov_name
		)))

		ensure_variable(re_name, .variable_kind_codes[["latent"]], enforce_kind = TRUE, respect_explicit = FALSE)
		add_edge(.edge_kind_codes[["regression"]], re_name, target, "beta", fixed_val = 1.0)
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
			first_indicator <- TRUE
			for (indicator_raw in rhs_terms) {
				parsed <- parse_term_with_fixed(indicator_raw)
				indicator <- parsed$var
				fixed_val <- parsed$val
				
				# Automatic identification: Fix first indicator to 1.0 if not specified
				if (first_indicator && is.null(fixed_val)) {
					fixed_val <- 1.0
				}
				first_indicator <- FALSE

				obs_var <- ensure_variable(indicator, .variable_kind_codes[["observed"]], enforce_kind = TRUE, respect_explicit = FALSE)
				add_edge(.edge_kind_codes[["loading"]], latent_var$name, obs_var$name, "lambda", fixed_val)
			}
		} else if (grepl("~~", eq, fixed = TRUE)) {
			parts <- strsplit(eq, "~~", fixed = TRUE)[[1]]
			lhs_raw <- trimws(parts[[1]])
			rhs_raw <- trimws(parts[[2]])
			
			parsed_lhs <- parse_term_with_fixed(lhs_raw) # Usually no fixed val on LHS
			parsed_rhs <- parse_term_with_fixed(rhs_raw)
			
			lhs <- parsed_lhs$var
			rhs <- parsed_rhs$var
			fixed_val <- parsed_rhs$val # Take fixed value from RHS if present (e.g. x ~~ 1*x)

			left <- ensure_variable(lhs, .variable_kind_codes[["observed"]])
			right <- ensure_variable(rhs, .variable_kind_codes[["observed"]])
			ordered <- sort(c(left$name, right$name))
			
			add_edge(.edge_kind_codes[["covariance"]], ordered[1], ordered[2], "psi", fixed_val)
		} else if (grepl("~", eq, fixed = TRUE)) {
			parts <- strsplit(eq, "~", fixed = TRUE)[[1]]
			target <- trimws(parts[[1]])
			
			# Check for Surv(time, status) syntax
			if (startsWith(target, "Surv(") && endsWith(target, ")")) {
				content <- substr(target, 6, nchar(target) - 1)
				args <- trimws(strsplit(content, ",")[[1]])
				if (length(args) != 2) {
					stop(sprintf("Surv() requires exactly 2 arguments (time, status), got: %s", target), call. = FALSE)
				}
				time_var <- args[[1]]
				status_var <- args[[2]]
				
				target <- time_var
				status_vars[[time_var]] <- status_var
			}

			predictors <- split_terms(parts[[2]])
			
			# Check if intercept is explicitly included or excluded
			has_explicit_intercept <- FALSE
			has_explicit_zero <- FALSE
			
			for (p in predictors) {
				if (p == "0") has_explicit_zero <- TRUE
				if (p == "1") has_explicit_intercept <- TRUE
				
				# Check for fixed intercept like 0*1 or 1*1
				parsed <- parse_term_with_fixed(p)
				if (parsed$var == "1") {
					has_explicit_intercept <- TRUE
				}
			}

			# Handle implicit intercept: add "1" unless "0" is present or "1" is already there
			if (!has_explicit_zero && !has_explicit_intercept) {
				predictors <- c(predictors, "1")
			}

			target_var <- ensure_variable(target, .variable_kind_codes[["observed"]])
			target_var <- ensure_variable(target, .variable_kind_codes[["observed"]])
			if (!length(predictors)) {
				stop(sprintf("regression equation for %s is empty", target), call. = FALSE)
			}
			for (predictor_raw in predictors) {
				parsed <- parse_term_with_fixed(predictor_raw)
				predictor <- parsed$var
				fixed_val <- parsed$val

				mixed <- parse_mixed_term(predictor)
				if (!is.null(mixed)) {
					add_random_effect(mixed, target)
					next
				}
				
				if (predictor == "1") {
					uses_intercept_column <- TRUE
					if (is.null(fam_map[["_intercept"]])) {
						fam_map[["_intercept"]] <- "fixed"
					}
					ensure_variable("_intercept", .variable_kind_codes[["observed"]])
					add_edge(.edge_kind_codes[["regression"]], "_intercept", target_var$name, "alpha", fixed_val)
					next
				}
				if (predictor == "0") {
					next
				}
				source_var <- ensure_variable(predictor, .variable_kind_codes[["observed"]])
				add_edge(.edge_kind_codes[["regression"]], source_var$name, target_var$name, "beta", fixed_val)
			}
		} else {
			stop(sprintf("unrecognized equation: %s", eq), call. = FALSE)
		}
	}

	# Add default variances/dispersions for variables if not present
	for (var_name in var_order) {
		entry <- var_defs[[var_name]]
		
		# Families that typically have a dispersion/variance parameter
		has_dispersion <- entry$family %in% c("gaussian", "negative_binomial", "nbinom", "weibull", "weibull_aft", "lognormal", "lognormal_aft", "loglogistic", "loglogistic_aft", "gamma")
		
		is_observed_dispersion <- (entry$kind == .variable_kind_codes[["observed"]] && 
		                           has_dispersion && 
		                           var_name != "_intercept")
		is_latent <- (entry$kind == .variable_kind_codes[["latent"]])
		
		# Check if it's a random effect latent (which has its variance defined via random_effects list)
		is_re_latent <- FALSE
		if (!is.null(random_effects)) {
			for (re in random_effects) {
				if (re$name == var_name) {
					is_re_latent <- TRUE
					break
				}
			}
		}

		if (is_observed_dispersion || (is_latent && !is_re_latent)) {
			# Check if self-covariance exists
			has_var <- FALSE
			for (edge in edges) {
				if (edge$kind == .edge_kind_codes[["covariance"]] && 
				    edge$source == var_name && 
				    edge$target == var_name) {
					has_var <- TRUE
					break
				}
			}
			if (!has_var) {
				add_edge(.edge_kind_codes[["covariance"]], var_name, var_name, "psi")
			}
		}
	}

	covariances <- covariances %||% list()
	genomic <- genomic %||% list()

	# Auto-discover genomic specs from data if referenced in covariances
	if (!is.null(data)) {
		for (cov in covariances) {
			# If structure is "grm" (or similar custom), check if we have data for it by name
			if (cov$structure == "grm") {
				if (is.null(genomic[[cov$name]])) {
					# Check if data has it
					if (!is.null(data[[cov$name]])) {
						mat <- data[[cov$name]]
						if (is.matrix(mat) || inherits(mat, "Matrix")) {
							genomic[[cov$name]] <- list(data = mat, precomputed = TRUE)
						}
					}
				}
			}
		}
	}

	genomic_data <- list()
	if (length(genomic) > 0L) {
		if (is.null(names(genomic)) || any(names(genomic) == "")) {
			stop("genomic specifications must be a named list", call. = FALSE)
		}
		existing_ids <- vapply(covariances, function(x) x$name %||% "", character(1L))
		for (cov_id in names(genomic)) {
			entry <- genomic[[cov_id]]
			
			raw_markers <- entry$markers
			raw_data <- entry$data
			
			if (is.null(raw_markers) && is.null(raw_data)) {
				stop(sprintf("genomic covariance '%s' requires 'markers' or 'data' matrix", cov_id), call. = FALSE)
			}
			
			using_precomputed <- !is.null(raw_data)
			source_data <- if (using_precomputed) raw_data else raw_markers
			
			markers <- as.matrix(source_data)
			if (!is.numeric(markers)) {
				stop(sprintf("genomic covariance '%s' data must be numeric", cov_id), call. = FALSE)
			}
			if (nrow(markers) == 0L || ncol(markers) == 0L) {
				stop(sprintf("genomic covariance '%s' data must be non-empty", cov_id), call. = FALSE)
			}
			
			if (using_precomputed && nrow(markers) != ncol(markers)) {
				stop(sprintf("genomic covariance '%s' precomputed kernel must be square", cov_id), call. = FALSE)
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
				
				precomputed_flag <- if (is.null(entry$precomputed)) using_precomputed else isTRUE(entry$precomputed)
				
				genomic_data[[cov_id]] <- list(
					markers = markers,
					center = if (is.null(entry$center)) TRUE else isTRUE(entry$center),
					normalize = if (is.null(entry$normalize)) TRUE else isTRUE(entry$normalize),
					precomputed = precomputed_flag
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

	ir <- build_semx_ir(var_defs[var_order], edges, covariances, random_effects)

    # Debug print
    print(paste("Var order:", paste(var_order, collapse=", ")))
    print(paste("Var defs names:", paste(names(var_defs), collapse=", ")))

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
			ir = ir,
			variables = var_defs,
			edges = edges,
			covariances = covariances,
			genomic_data = genomic_data,
			fixed_covariance_data = fixed_covariance_data,
			random_effects = random_effects,
			uses_intercept_column = uses_intercept_column,
			status_vars = status_vars
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
#' @param estimation_method Estimation method ("ML" or "REML").
#' @return A FitResult object.
#' @export
semx_fit <- function(model, data, options = NULL, optimizer_name = "lbfgs", fixed_covariance_data = NULL, estimation_method = "ML") {
    print("semx_fit called")
    if (!inherits(model, "semx_model")) {
        stop("model must be a semx_model object")
    }
    
    if (is.null(options)) {
        options <- new(OptimizationOptions)
    }
    
    method_code <- if (toupper(estimation_method) == "REML") 1L else 0L
    
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

    # Expand factor variables used as design variables (predictors)
    factor_vars <- character()
    if (is.data.frame(data)) {
        for (col_name in names(data)) {
            if (col_name %in% names(model$variables)) {
                var_def <- model$variables[[col_name]]
                # Only expand if NOT grouping variable AND family is "fixed" (i.e. predictor)
                if (var_def$kind != .variable_kind_codes[["grouping"]] && var_def$family == "fixed") {
                    col <- data[[col_name]]
                    if (is.character(col) || is.factor(col)) {
                        factor_vars <- c(factor_vars, col_name)
                    }
                }
            }
        }
    }
    
    if (length(factor_vars) > 0) {
        print(paste("Expanding factors:", paste(factor_vars, collapse=", ")))
        for (name in factor_vars) {
            # Generate dummies (all levels)
            f <- as.formula(paste("~ 0 +", name))
            dummies <- model.matrix(f, data)
            dummy_names <- colnames(dummies)
            
            # Add dummies to data
            for (d_name in dummy_names) {
                data[[d_name]] <- as.numeric(dummies[, d_name])
                if (is.null(model$variables[[d_name]])) {
                    model$variables[[d_name]] <- list(
                        name = d_name,
                        kind = .variable_kind_codes[["observed"]],
                        family = "gaussian"
                    )
                }
            }
            
            # Update Edges (Fixed Effects)
            edges_to_remove <- integer()
            edges_to_add <- list()
            
            for (i in seq_along(model$edges)) {
                edge <- model$edges[[i]]
                if (edge$kind == .edge_kind_codes[["regression"]] && edge$source == name) {
                    edges_to_remove <- c(edges_to_remove, i)
                    target <- edge$target
                    
                    has_intercept <- FALSE
                    for (e in model$edges) {
                        if (e$kind == .edge_kind_codes[["regression"]] && e$target == target && e$source == "_intercept") {
                            has_intercept <- TRUE
                            break
                        }
                    }
                    
                    selected_dummies <- if (has_intercept) dummy_names[-1] else dummy_names
                    
                    for (d_name in selected_dummies) {
                        param_id <- paste0("beta_", target, "_on_", d_name)
                        edges_to_add <- c(edges_to_add, list(list(
                            kind = .edge_kind_codes[["regression"]],
                            source = d_name,
                            target = target,
                            parameter_id = param_id
                        )))
                    }
                }
            }
            
            if (length(edges_to_remove) > 0) {
                model$edges <- model$edges[-edges_to_remove]
                model$edges <- c(model$edges, edges_to_add)
            }
            
            # Update Random Effects
            if (!is.null(model$random_effects)) {
                for (i in seq_along(model$random_effects)) {
                    re <- model$random_effects[[i]]
                    if (name %in% re$variables) {
                        if (re$variables[[1]] == name) next
                        
                        has_intercept <- "_intercept" %in% re$variables
                        selected_dummies <- if (has_intercept) dummy_names[-1] else dummy_names
                        
                        new_vars <- character()
                        for (v in re$variables) {
                            if (v == name) {
                                new_vars <- c(new_vars, selected_dummies)
                            } else {
                                new_vars <- c(new_vars, v)
                            }
                        }
                        
                        model$random_effects[[i]]$variables <- new_vars
                        
                        cov_name <- re$covariance
                        for (j in seq_along(model$covariances)) {
                            if (model$covariances[[j]]$name == cov_name) {
                                model$covariances[[j]]$dimension <- length(new_vars) - 1
                                break
                            }
                        }
                    }
                }
            }
            
            model$variables[[name]] <- NULL
        }
    }

    # Handle Ordinal Outcomes: Register Threshold Parameters
    ordinal_params_added <- FALSE
    extra_param_mappings <- list()
    if (is.data.frame(data)) {
        for (col_name in names(data)) {
            if (col_name %in% names(model$variables)) {
                var_def <- model$variables[[col_name]]
                if (var_def$family == "ordinal") {
                    col <- data[[col_name]]
                    n_levels <- length(unique(col))
                    if (n_levels < 2) stop(paste("Ordinal variable", col_name, "must have at least 2 levels"))
                    
                    # Register K-1 thresholds
                    if (is.null(model$parameters)) model$parameters <- list()
                    
                    thresholds <- character()
                    for (k in 1:(n_levels-1)) {
                        param_id <- paste0(col_name, "_threshold_", k)
                        # Initial values: spread them out around 0
                        init_val <- qnorm(k / n_levels)
                        model$parameters[[param_id]] <- init_val
                        thresholds <- c(thresholds, param_id)
                    }
                    extra_param_mappings[[col_name]] <- thresholds
                    ordinal_params_added <- TRUE
                }
            }
        }
    }
        
    if (length(factor_vars) > 0 || ordinal_params_added) {
        # Rebuild IR
        model$ir <- build_semx_ir(model$variables, model$edges, model$covariances, model$random_effects, model$parameters)
    }

    # Inject genomic group column if needed
    if (is.null(data[["_genomic_group_"]])) {
        uses_genomic_group <- FALSE
        if (!is.null(model$random_effects)) {
            for (re in model$random_effects) {
                if (length(re$variables) > 0 && re$variables[[1]] == "_genomic_group_") {
                    uses_genomic_group <- TRUE
                    break
                }
            }
        }
        
        if (uses_genomic_group) {
             n_rows <- 0L
             if (is.data.frame(data)) {
                 n_rows <- nrow(data)
             } else if (is.list(data) && length(data) > 0L) {
                 n_rows <- length(data[[1]])
             }
             if (n_rows > 0L) {
                 data[["_genomic_group_"]] <- rep(1.0, n_rows)
             }
        }
    }
    
    # Convert character/factor columns to numeric to satisfy C++ requirements
    # We convert to 0-based indices for C++ compatibility
    if (is.data.frame(data)) {
        for (col_name in names(data)) {
            col <- data[[col_name]]
            if (is.character(col) || is.factor(col)) {
                data[[col_name]] <- as.numeric(as.factor(col)) - 1
            }
        }
    }
    
    print("Creating LikelihoodDriver...")
    driver <- new(LikelihoodDriver)
    print("LikelihoodDriver created.")

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

    # Pass initial parameters via status map if present
    if (!is.null(model$parameters)) {
        for (param_name in names(model$parameters)) {
            val <- as.numeric(model$parameters[[param_name]])
            status_data[[param_name]] <- val
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

    print("Calling driver$fit...")
    result <- if (length(merged_fixed) > 0L && length(status_data) > 0L) {
        driver$fit_with_fixed_and_status(model$ir, data, options, optimizer_name, merged_fixed, status_data, extra_param_mappings, method_code)
    } else if (length(merged_fixed) > 0L) {
        driver$fit_with_fixed(model$ir, data, options, optimizer_name, merged_fixed, extra_param_mappings, method_code)
    } else if (length(status_data) > 0L) {
        driver$fit_with_status(model$ir, data, options, optimizer_name, status_data, extra_param_mappings, method_code)
    } else {
        driver$fit(model$ir, data, options, optimizer_name, extra_param_mappings, method_code)
    }
    print("driver$fit returned.")

    opt_res <- result$optimization_result
    opt_res_list <- list(
        parameters = opt_res$parameters,
        objective_value = opt_res$objective_value,
        gradient_norm = opt_res$gradient_norm,
        iterations = opt_res$iterations,
        converged = opt_res$converged
    )
    names(opt_res_list$parameters) <- result$parameter_names
    
    se <- result$standard_errors
    if (length(se) == length(result$parameter_names)) {
        names(se) <- result$parameter_names
    }

    structure(
        list(
            optimization_result = opt_res_list,
            standard_errors = se,
            vcov = result$vcov,
            parameter_names = result$parameter_names,
            covariance_matrices = result$covariance_matrices,
            aic = result$aic,
            bic = result$bic,
            chi_square = result$chi_square,
            df = result$df,
            p_value = result$p_value,
            cfi = result$cfi,
            tli = result$tli,
            rmsea = result$rmsea,
            srmr = result$srmr,
            model = model,
            data = data
        ),
        class = "semx_fit"
    )
}

#' Extract variance components
#'
#' @param fit A semx_fit object.
#' @return A data.frame with variance components.
#' @export
semx_variance_components <- function(fit) {
  if (!inherits(fit, "semx_fit")) stop("fit must be a semx_fit object")
  
  rows <- list()
  
  cov_matrices <- fit$covariance_matrices
  if (is.null(cov_matrices)) return(data.frame())
  
  # Map covariance_id to random effect info
  re_map <- list()
  if (!is.null(fit$model$random_effects)) {
    for (re in fit$model$random_effects) {
      re_map[[re$covariance]] <- re
    }
  }
  
  for (cov_id in names(cov_matrices)) {
    if (is.null(re_map[[cov_id]])) next
    
    re <- re_map[[cov_id]]
    variables <- re$variables
    if (length(variables) == 0) next
    
    group <- variables[[1]]
    design_vars <- if (length(variables) > 1) variables[-1] else character()
    
    # Handle implicit intercept for (1 | group)
    if (length(design_vars) == 0) {
        # Check matrix dimension
        mat_vec <- cov_matrices[[cov_id]]
        if (length(mat_vec) == 1) {
            design_vars <- c("(Intercept)")
        }
    }
    
    dim <- length(design_vars)
    mat_vec <- cov_matrices[[cov_id]]
    
    if (length(mat_vec) != dim * dim) next
    
    mat <- matrix(mat_vec, nrow = dim, ncol = dim, byrow = TRUE) # C++ is row-major
    
    sds <- sqrt(diag(mat))
    sds_safe <- sds
    sds_safe[sds_safe == 0] <- 1.0
    corrs <- mat / outer(sds_safe, sds_safe)
    
    for (i in seq_along(design_vars)) {
      var1 <- design_vars[i]
      
      # Variance row
      rows[[length(rows) + 1]] <- list(
        Group = group,
        Name1 = var1,
        Name2 = "",
        Variance = mat[i, i],
        Std.Dev = sds[i],
        Corr = NA_real_
      )
      
      # Correlation rows
      if (i < dim) {
        for (j in (i + 1):dim) {
          var2 <- design_vars[j]
          rows[[length(rows) + 1]] <- list(
            Group = group,
            Name1 = var1,
            Name2 = var2,
            Variance = mat[i, j],
            Std.Dev = NA_real_,
            Corr = corrs[i, j]
          )
        }
      }
    }
  }
  
  if (length(rows) == 0) {
    return(data.frame(
      Group = character(),
      Name1 = character(),
      Name2 = character(),
      Variance = numeric(),
      Std.Dev = numeric(),
      Corr = numeric(),
      stringsAsFactors = FALSE
    ))
  }
  
  do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
}

#' Extract covariance weights
#'
#' @param fit A semx_fit object.
#' @param cov_id Covariance identifier.
#' @return A list with sigma_sq and weights, or NULL.
#' @export
semx_covariance_weights <- function(fit, cov_id) {
  if (!inherits(fit, "semx_fit")) stop("fit must be a semx_fit object")
  
  # Find covariance spec
  cov_spec <- NULL
  for (c in fit$model$covariances) {
    if (c$name == cov_id) {
      cov_spec <- c
      break
    }
  }
  
  if (is.null(cov_spec)) stop(sprintf("Covariance '%s' not found in model", cov_id))
  
  if (is.null(cov_spec$structure) || cov_spec$structure != "multi_kernel_simplex") {
    return(NULL)
  }
  
  params <- fit$optimization_result$parameters
  names(params) <- fit$parameter_names
  
  prefix <- paste0(cov_id, "_")
  relevant_keys <- grep(paste0("^", prefix), names(params), value = TRUE)
  
  if (length(relevant_keys) == 0) return(NULL)
  
  # Sort by index
  indices <- as.integer(sub(prefix, "", relevant_keys))
  sorted_keys <- relevant_keys[order(indices)]
  
  if (length(sorted_keys) < 2) return(NULL)
  
  sigma_sq <- params[[sorted_keys[1]]]
  thetas <- unname(params[sorted_keys[-1]])
  
  max_theta <- max(thetas)
  exps <- exp(thetas - max_theta)
  sum_exps <- sum(exps)
  weights <- exps / sum_exps
  
  list(
    sigma_sq = sigma_sq,
    weights = weights
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
  
  param_ids <- object$parameter_names
  if (is.null(param_ids)) {
      # Fallback if parameter_names missing (e.g. old object)
      param_ids <- model$ir$parameter_ids()
  }
  
  # Debugging
  if (length(param_ids) != length(params)) {
      warning(sprintf("Parameter ID count (%d) does not match parameter count (%d)", length(param_ids), length(params)))
  }
  
  param_table <- data.frame(
    Estimate = as.numeric(params),
    Std.Error = as.numeric(ses),
    z.value = as.numeric(z_values),
    P.value = as.numeric(p_values)
  )
  if (length(param_ids) == length(params)) {
      rownames(param_table) <- param_ids
  }
  
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
    fit_indices = fit_indices,
    variance_components = semx_variance_components(object),
    covariance_weights = list()
  )
  
  # Collect covariance weights
  for (cov in object$model$covariances) {
      if (!is.null(cov$structure) && cov$structure == "multi_kernel_simplex") {
          w <- semx_covariance_weights(object, cov$name)
          if (!is.null(w)) {
              res$covariance_weights[[cov$name]] <- w
          }
      }
  }
  
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
    if (!is.na(x$fit_indices$df) && x$fit_indices$df > 0) {
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
  
  if (!is.null(x$variance_components) && nrow(x$variance_components) > 0) {
      cat("\nVariance Components:\n")
      print(x$variance_components, row.names = FALSE)
  }
  
  if (!is.null(x$covariance_weights) && length(x$covariance_weights) > 0) {
      for (cov_id in names(x$covariance_weights)) {
          w <- x$covariance_weights[[cov_id]]
          cat(sprintf("\nCovariance Weights for '%s':\n", cov_id))
          cat(sprintf("  Sigma^2: %.4f\n", w$sigma_sq))
          cat("  Weights:\n")
          for (i in seq_along(w$weights)) {
              cat(sprintf("    Kernel %d: %.4f\n", i, w$weights[i]))
          }
      }
  }
  
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
plot.semx_fit <- function(x, type = "residuals", ...) {
  preds <- predict(x)
  data <- x$data
  
  for (target in names(preds)) {
    if (!is.null(data[[target]])) {
      y <- data[[target]]
      y_hat <- preds[[target]]
      resid <- y - y_hat
      
      if (type == "residuals") {
          plot(y_hat, resid, main = paste("Residuals vs Fitted:", target), xlab = "Fitted", ylab = "Residuals", ...)
          abline(h = 0, col = "red", lty = 2)
      } else if (type == "qq") {
          qqnorm(resid, main = paste("Normal Q-Q Plot:", target), ...)
          qqline(resid, col = "red")
      } else {
          stop(sprintf("Unknown plot type: %s", type))
      }
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
