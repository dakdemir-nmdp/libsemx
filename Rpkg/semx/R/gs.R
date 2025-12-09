#' Genomic Selection Model
#'
#' Fits a genomic selection model (GBLUP) using a mixed model framework.
#'
#' @param formula A formula specifying the fixed effects (e.g., `y ~ 1 + cov1`).
#' @param data A data frame containing the variables in the formula.
#' @param geno_id Character string specifying the column in `data` that links observations to genotypes.
#' @param markers A numeric matrix of markers (rows = genotypes, cols = markers). Row names must match `geno_id` levels.
#' @param kernel Optional pre-computed relationship matrix (GRM). If provided, `markers` is ignored.
#' @param estimator Estimation method: "reml" (default) or "ml".
#' @param family Outcome distribution family (default: "gaussian").
#' @param ... Additional arguments passed to `semx_fit`.
#'
#' @return A `semx_fit` object.
#' @export
semx_gs <- function(formula, data, geno_id, markers = NULL, kernel = NULL, estimator = "reml", family = "gaussian", ...) {
  # Check inputs
  if (is.null(markers) && is.null(kernel)) {
    stop("Either 'markers' or 'kernel' must be provided.")
  }
  
  # Define covariance name
  cov_name <- "cov_g"
  
  # Extract terms from formula
  tf <- terms(as.formula(formula))
  vars <- attr(tf, "variables")
  resp_idx <- attr(tf, "response")
  
  if (resp_idx == 0) stop("Formula must have a response variable.")
  
  resp_name <- as.character(vars[[resp_idx + 1]]) # +1 because first is 'list'
  
  # Get RHS terms
  term_labels <- attr(tf, "term.labels")
  intercept <- attr(tf, "intercept")
  
  rhs_parts <- term_labels
  if (intercept == 0) {
    rhs_parts <- c("0", rhs_parts)
  }
  
  # Add genomic random effect
  # Syntax: (1 | genomic(geno_id, cov_name))
  re_term <- sprintf("(1 | genomic(%s, %s))", geno_id, cov_name)
  rhs_parts <- c(rhs_parts, re_term)
  
  eq_str <- paste(resp_name, "~", paste(rhs_parts, collapse = " + "))
  
  # Prepare genomic list
  genomic_list <- list()
  if (!is.null(kernel)) {
    genomic_list[[cov_name]] <- list(data = kernel, precomputed = TRUE)
  } else {
    genomic_list[[cov_name]] <- list(markers = markers)
  }
  
  # Prepare families
  families <- setNames(family, resp_name)
  # families[[geno_id]] <- "index" # Removed as it causes "Unknown outcome family" error
  
  # Call semx_fit
  semx_fit(
    model = semx_model(
      equations = eq_str,
      families = families,
      genomic = genomic_list,
      data = data # Pass data to semx_model to resolve names if needed
    ),
    data = data,
    estimation_method = estimator,
    ...
  )
}
