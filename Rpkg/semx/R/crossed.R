#' Fit Crossed Random Effects Model using Method of Moments
#'
#' Fits a linear mixed model with two crossed random effects using the Method of Moments (MoM).
#' This provides fast O(N) estimation for models of the form:
#'
#' \deqn{y = X\beta + Z_u u + Z_v v + e}
#'
#' where u ~ N(0, σ²_u I), v ~ N(0, σ²_v I), and e ~ N(0, σ²_e I).
#'
#' The MoM approach uses U-statistics (Gao & Owen 2017, 2018) to estimate variance components
#' without iterative optimization, making it ideal for large-scale datasets.
#'
#' @param formula Model formula for fixed effects (e.g., y ~ x1 + x2).
#' @param data Data frame containing the variables.
#' @param u Name of the column for the first random effect grouping factor.
#' @param v Name of the column for the second random effect grouping factor.
#' @param use_gls Logical; if TRUE, refine fixed effects using GLS after initial OLS estimation (default: FALSE).
#' @param second_step Logical; if TRUE, perform a second MoM step using GLS residuals (default: FALSE).
#' @param verbose Logical; if TRUE, print convergence details (default: FALSE).
#' @param min_variance Numeric; minimum variance floor for variance components (default: 1e-10).
#'
#' @return An object of class \code{semx_crossed_fit} with the following components:
#' \describe{
#'   \item{beta}{Fixed effects estimates.}
#'   \item{variance_components}{Variance component estimates c(σ²_u, σ²_v, σ²_e).}
#'   \item{n_groups_u}{Number of unique levels for first random effect.}
#'   \item{n_groups_v}{Number of unique levels for second random effect.}
#'   \item{converged}{Logical indicating convergence.}
#'   \item{message}{Convergence message.}
#'   \item{fixed_names}{Names of fixed effects.}
#'   \item{u_name}{Name of first random grouping factor.}
#'   \item{v_name}{Name of second random grouping factor.}
#' }
#'
#' @examples
#' \dontrun{
#' # Simulate data with crossed random effects
#' set.seed(42)
#' n_students <- 30
#' n_schools <- 15
#' n_reps <- 4
#'
#' data <- expand.grid(
#'   student = 1:n_students,
#'   school = 1:n_schools,
#'   rep = 1:n_reps
#' )
#' data$x <- rnorm(nrow(data))
#' data$y <- 5.0 + 1.5 * data$x +
#'   rnorm(n_students)[data$student] * sqrt(0.6) +  # student effects
#'   rnorm(n_schools)[data$school] * sqrt(0.4) +    # school effects
#'   rnorm(nrow(data)) * sqrt(0.3)                  # residual
#'
#' # Fit model
#' fit <- semx_crossed(y ~ 1 + x, data, u = "student", v = "school")
#' print(fit)
#' summary(fit)
#' }
#'
#' @references
#' Gao, K., & Owen, A. B. (2017). Efficient moment calculations for variance components
#' in large unbalanced crossed random effects models. Electronic Journal of Statistics,
#' 11(1), 1235-1296.
#'
#' Gao, K., & Owen, A. B. (2018). Estimation and inference for very large linear mixed
#' effects models. Statistica Sinica, 30(3), 1741-1771.
#'
#' @export
semx_crossed <- function(formula, data, u, v,
                         use_gls = FALSE,
                         second_step = FALSE,
                         verbose = FALSE,
                         min_variance = 1e-10) {

  # Parse formula
  if (!inherits(formula, "formula")) {
    stop("formula must be a formula object")
  }

  # Extract response and predictors
  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- model.matrix(formula, mf)
  fixed_names <- colnames(X)

  # Check grouping variables
  if (!u %in% colnames(data)) {
    stop(sprintf("Grouping variable '%s' not found in data", u))
  }
  if (!v %in% colnames(data)) {
    stop(sprintf("Grouping variable '%s' not found in data", v))
  }

  # Convert grouping factors to integer indices (0-based for C++)
  u_factor <- as.factor(data[[u]])
  v_factor <- as.factor(data[[v]])

  u_indices <- as.integer(u_factor) - 1L  # 0-based indexing
  v_indices <- as.integer(v_factor) - 1L  # 0-based indexing

  # Fit model
  result <- mom_solver_fit(y, X, u_indices, v_indices,
                          use_gls, second_step, verbose, min_variance)

  # Create result object
  fit <- list(
    beta = as.vector(result$beta),
    variance_components = as.vector(result$variance_components),
    n_groups_u = result$n_groups_u,
    n_groups_v = result$n_groups_v,
    converged = result$converged,
    message = result$message,
    fixed_names = fixed_names,
    u_name = u,
    v_name = v,
    formula = formula,
    data_name = deparse(substitute(data))
  )

  class(fit) <- "semx_crossed_fit"
  return(fit)
}

#' Print method for semx_crossed_fit
#'
#' @param x An object of class \code{semx_crossed_fit}.
#' @param ... Additional arguments (ignored).
#'
#' @export
print.semx_crossed_fit <- function(x, ...) {
  cat("Crossed Random Effects Model (Method of Moments)\n")
  cat(sprintf("Formula: %s\n", deparse(x$formula)))
  cat(sprintf("Converged: %s\n", x$converged))
  if (!x$converged) {
    cat(sprintf("Message: %s\n", x$message))
  }
  cat("\n")
  cat("Fixed Effects:\n")
  for (i in seq_along(x$fixed_names)) {
    cat(sprintf("  %-20s %10.4f\n", x$fixed_names[i], x$beta[i]))
  }
  cat("\n")
  cat("Variance Components:\n")
  cat(sprintf("  σ²_%-17s %10.4f\n", x$u_name, x$variance_components[1]))
  cat(sprintf("  σ²_%-17s %10.4f\n", x$v_name, x$variance_components[2]))
  cat(sprintf("  σ²_residual        %10.4f\n", x$variance_components[3]))
  cat("\n")
  cat(sprintf("Number of groups: %s=%d, %s=%d\n",
              x$u_name, x$n_groups_u,
              x$v_name, x$n_groups_v))
  invisible(x)
}

#' Summary method for semx_crossed_fit
#'
#' @param object An object of class \code{semx_crossed_fit}.
#' @param ... Additional arguments (ignored).
#'
#' @export
summary.semx_crossed_fit <- function(object, ...) {
  cat("=====================================================================\n")
  cat("Crossed Random Effects Model (Method of Moments)\n")
  cat("=====================================================================\n\n")

  cat(sprintf("Formula: %s\n", deparse(object$formula)))
  cat(sprintf("Data: %s\n\n", object$data_name))

  cat(sprintf("Converged: %s\n", object$converged))
  if (!object$converged) {
    cat(sprintf("Message: %s\n", object$message))
  }
  cat("\n")

  cat("Fixed Effects:\n")
  cat("---------------------------------------------------------------------\n")
  cat(sprintf("%-20s %10s\n", "Parameter", "Estimate"))
  cat("---------------------------------------------------------------------\n")
  for (i in seq_along(object$fixed_names)) {
    cat(sprintf("%-20s %10.4f\n", object$fixed_names[i], object$beta[i]))
  }
  cat("---------------------------------------------------------------------\n\n")

  cat("Variance Components:\n")
  cat("---------------------------------------------------------------------\n")
  cat(sprintf("%-20s %10s\n", "Component", "Estimate"))
  cat("---------------------------------------------------------------------\n")
  cat(sprintf("σ²_%-17s %10.4f\n", object$u_name, object$variance_components[1]))
  cat(sprintf("σ²_%-17s %10.4f\n", object$v_name, object$variance_components[2]))
  cat(sprintf("σ²_residual        %10.4f\n", object$variance_components[3]))
  total_var <- sum(object$variance_components)
  cat("---------------------------------------------------------------------\n")
  cat(sprintf("Total Variance:     %10.4f\n", total_var))
  cat("---------------------------------------------------------------------\n\n")

  cat(sprintf("Sample size: N = %d\n", object$n_groups_u * object$n_groups_v))
  cat(sprintf("Groups: %s = %d, %s = %d\n",
              object$u_name, object$n_groups_u,
              object$v_name, object$n_groups_v))
  cat("\n")

  cat("=====================================================================\n")
  cat("Method of Moments uses U-statistics for O(N) variance estimation.\n")
  cat("See Gao & Owen (2017, 2018) for details.\n")
  cat("=====================================================================\n")

  invisible(object)
}
