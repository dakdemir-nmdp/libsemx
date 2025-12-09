#' Genotype-by-Environment (GxE) Model
#'
#' Fits a Genotype-by-Environment (GxE) model using a mixed model framework.
#'
#' This function simplifies the construction of GxE models, supporting both
#' standard diagonal (independent) covariance structures and Kronecker product
#' structures ($K_G \otimes I_E$) for the interaction term.
#'
#' @param formula A formula specifying the fixed effects (e.g., `yield ~ 1`).
#' @param data A data frame containing the variables.
#' @param genotype Character string specifying the genotype column name.
#' @param environment Character string specifying the environment column name.
#' @param genomic Optional numeric matrix representing the genomic relationship matrix (GRM) or markers.
#'   If provided, the genotype random effect will use this covariance.
#' @param gxe_structure Character string: "diagonal" (default) or "kronecker".
#'   "kronecker" requires `genomic` to be provided.
#' @param family Outcome distribution family (default: "gaussian").
#' @param ... Additional arguments passed to `semx_fit`.
#'
#' @return A `semx_fit` object.
#' @export
semx_gxe <- function(formula, data, genotype, environment, genomic = NULL, gxe_structure = "diagonal", family = "gaussian", ...) {
  
  # Validate inputs
  if (!genotype %in% names(data)) stop(sprintf("Column '%s' not found in data", genotype))
  if (!environment %in% names(data)) stop(sprintf("Column '%s' not found in data", environment))
  
  # Parse formula to get response
  tf <- terms(as.formula(formula))
  vars <- attr(tf, "variables")
  resp_idx <- attr(tf, "response")
  if (resp_idx == 0) stop("Formula must have a response variable.")
  response <- as.character(vars[[resp_idx + 1]])
  
  # Prepare data copy
  df <- data
  
  # Factorize grouping variables to 0-based integers for C++
  # We use factor() to ensure consistent ordering
  
  # Genotype
  if (!is.numeric(df[[genotype]])) {
    f_g <- factor(df[[genotype]])
    # Sort levels if they are not ordered? 
    # Python implementation used sort=True. R factor levels are sorted by default if character.
    # But if they are already factor, we keep levels.
    # To match Python behavior (and potentially sorted GRM), we might want to force sort.
    # But user might have provided GRM matching the factor levels.
    # Let's assume user knows what they are doing with factor levels vs GRM rows.
    df[[genotype]] <- as.integer(f_g) - 1L
    n_g <- nlevels(f_g)
  } else {
    # Already numeric? Assume 0-based or 1-based?
    # C++ expects 0-based.
    # If user passes integers, we assume they are indices.
    # But safer to re-factorize to be sure.
    f_g <- factor(df[[genotype]])
    df[[genotype]] <- as.integer(f_g) - 1L
    n_g <- nlevels(f_g)
  }
  
  # Environment
  if (!is.numeric(df[[environment]])) {
    f_e <- factor(df[[environment]])
    df[[environment]] <- as.integer(f_e) - 1L
    n_e <- nlevels(f_e)
  } else {
    f_e <- factor(df[[environment]])
    df[[environment]] <- as.integer(f_e) - 1L
    n_e <- nlevels(f_e)
  }
  
  # Interaction
  interaction_col <- paste0(genotype, "_x_", environment)
  
  # Lists for semx_model
  covariances <- list()
  random_effects <- list()
  genomic_args <- list()
  
  # 1. Genotype Random Effect
  re_g_name <- paste0("u_", genotype)
  
  if (!is.null(genomic)) {
    cov_name_g <- "K_g"
    genomic_args[[cov_name_g]] <- list(markers = genomic)
    # Note: semx_model handles adding this to covariances if in genomic args?
    # Actually semx_model implementation adds genomic entries to covariances list.
    # So we don't need to add it manually to covariances list.
    
    re_g <- list(
      name = re_g_name,
      variables = c(genotype, "_intercept"),
      covariance = cov_name_g
    )
  } else {
    cov_name_g <- paste0("cov_", genotype)
    covariances[[length(covariances) + 1]] <- list(
      name = cov_name_g,
      structure = "diagonal",
      dimension = 1L
    )
    re_g <- list(
      name = re_g_name,
      variables = c(genotype, "_intercept"),
      covariance = cov_name_g
    )
  }
  random_effects[[length(random_effects) + 1]] <- re_g
  
  # 2. Environment Random Effect
  cov_name_e <- paste0("cov_", environment)
  covariances[[length(covariances) + 1]] <- list(
    name = cov_name_e,
    structure = "diagonal",
    dimension = 1L
  )
  re_e <- list(
    name = paste0("u_", environment),
    variables = c(environment, "_intercept"),
    covariance = cov_name_e
  )
  random_effects[[length(random_effects) + 1]] <- re_e
  
  # 3. Interaction Random Effect
  if (gxe_structure == "kronecker") {
    if (is.null(genomic)) stop("Kronecker GxE structure requires 'genomic' matrix.")
    
    # Create interaction index: g * n_e + e
    # Assuming K_g is outer, I_e is inner.
    # df[[genotype]] and df[[environment]] are 0-based.
    df[[interaction_col]] <- df[[genotype]] * n_e + df[[environment]]
    
    cov_name_ge <- "K_ge"
    cov_name_ie <- "I_e"
    
    # Define I_e
    covariances[[length(covariances) + 1]] <- list(
      name = cov_name_ie,
      structure = "identity",
      dimension = n_e
    )
    
    # Define K_ge
    covariances[[length(covariances) + 1]] <- list(
      name = cov_name_ge,
      structure = "kronecker",
      dimension = n_g * n_e,
      components = list(
          list(id = "K_g"),
          list(id = cov_name_ie)
      )
    )
    
    re_ge <- list(
      name = paste0("u_", interaction_col),
      variables = c(interaction_col, "_intercept"),
      covariance = cov_name_ge
    )
    random_effects[[length(random_effects) + 1]] <- re_ge
    
  } else {
    # Diagonal
    # Create interaction factor
    int_fac <- factor(paste(df[[genotype]], df[[environment]], sep = ":"))
    df[[interaction_col]] <- as.integer(int_fac) - 1L
    
    cov_name_gxe <- paste0("cov_", interaction_col)
    covariances[[length(covariances) + 1]] <- list(
      name = cov_name_gxe,
      structure = "diagonal",
      dimension = 1L
    )
    
    re_gxe <- list(
      name = paste0("u_", interaction_col),
      variables = c(interaction_col, "_intercept"),
      covariance = cov_name_gxe
    )
    random_effects[[length(random_effects) + 1]] <- re_gxe
  }
  
  # Prepare families
  families <- setNames(family, response)
  # Add intercept family if needed? semx_model handles it if we pass families.
  # But we need to ensure _intercept is registered if used.
  # semx_model does this if it sees _intercept in variables.
  
  # Construct model
  model <- semx_model(
    equations = c(formula),
    families = families,
    covariances = covariances,
    random_effects = random_effects,
    genomic = genomic_args,
    data = df
  )
  
  # Fit model
  fit <- semx_fit(model, df, ...)
  
  return(fit)
}
