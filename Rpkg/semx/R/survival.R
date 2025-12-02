#' Predict survival probabilities
#'
#' @param fit A semx_fit object.
#' @param newdata A data.frame.
#' @param times Numeric vector of times.
#' @param outcome Name of the outcome variable.
#' @return A data.frame of survival probabilities.
#' @export
semx_predict_survival <- function(fit, newdata, times, outcome) {
  model <- fit$model
  params <- fit$optimization_result$parameters
  names(params) <- model$ir$parameter_ids()
  
  # Find family
  var_def <- model$variables[[outcome]]
  if (is.null(var_def)) stop(sprintf("Outcome '%s' not found", outcome))
  family <- var_def$family
  
  # Predict eta
  preds <- predict(fit, newdata)
  if (is.null(preds[[outcome]])) stop(sprintf("Could not predict for '%s'", outcome))
  eta <- preds[[outcome]]
  
  # Find dispersion
  dispersion_id <- sprintf("psi_%s_%s", outcome, outcome)
  k <- params[[dispersion_id]]
  
  if (is.null(k)) {
    if (family == "exponential") {
      k <- 1.0
    } else {
      # Fuzzy search
      candidates <- grep(paste0("psi.*", outcome, "|shape.*", outcome), names(params), value = TRUE)
      if (length(candidates) == 1) {
        k <- params[[candidates]]
      } else {
        stop(sprintf("Dispersion parameter for '%s' not found", outcome))
      }
    }
  }
  
  results <- matrix(NA, nrow = length(eta), ncol = length(times))
  colnames(results) <- as.character(times)
  
  for (j in seq_along(times)) {
    t <- times[j]
    if (t < 0) next
    
    if (family %in% c("weibull", "exponential")) {
      log_z <- k * (log(t) - eta)
      z <- exp(log_z)
      S_t <- exp(-z)
    } else if (family == "loglogistic") {
      log_z <- k * (log(t) - eta)
      z <- exp(log_z)
      S_t <- 1.0 / (1.0 + z)
    } else if (family == "lognormal") {
      z_score <- (log(t) - eta) / k
      S_t <- 1.0 - pnorm(z_score)
    } else {
      stop(sprintf("Unsupported family: %s", family))
    }
    results[, j] <- S_t
  }
  
  as.data.frame(results)
}

#' Predict Cumulative Incidence Function
#'
#' @param fit A semx_fit object.
#' @param newdata A data.frame.
#' @param times Numeric vector of times.
#' @param event_outcome Name of the event outcome.
#' @param competing_outcomes Vector of competing outcome names.
#' @return A data.frame of CIF values.
#' @export
semx_predict_cif <- function(fit, newdata, times, event_outcome, competing_outcomes) {
  all_outcomes <- c(event_outcome, competing_outcomes)
  preds <- predict(fit, newdata)
  n_obs <- nrow(newdata)
  
  params <- fit$optimization_result$parameters
  names(params) <- fit$model$ir$parameter_ids()
  
  outcome_params <- list()
  for (out in all_outcomes) {
    var_def <- fit$model$variables[[out]]
    family <- var_def$family
    eta <- preds[[out]]
    
    dispersion_id <- sprintf("psi_%s_%s", out, out)
    k <- params[[dispersion_id]]
    if (is.null(k)) {
      if (family == "exponential") k <- 1.0
      else {
         candidates <- grep(paste0("psi.*", out, "|shape.*", out), names(params), value = TRUE)
         if (length(candidates) == 1) k <- params[[candidates]]
         else stop(sprintf("Dispersion for %s not found", out))
      }
    }
    outcome_params[[out]] <- list(eta = eta, k = k, family = family)
  }
  
  sorted_times <- sort(unique(c(0, times[times >= 0])))
  cif_values <- matrix(0, nrow = n_obs, ncol = length(sorted_times))
  
  for (i in 2:length(sorted_times)) {
    t_mid <- (sorted_times[i-1] + sorted_times[i]) / 2
    dt <- sorted_times[i] - sorted_times[i-1]
    
    S_overall <- rep(1, n_obs)
    for (out in all_outcomes) {
      p <- outcome_params[[out]]
      k <- p$k; eta <- p$eta; fam <- p$family
      
      if (fam %in% c("weibull", "exponential")) {
        z <- exp(k * (log(t_mid) - eta))
        S_j <- exp(-z)
      } else if (fam == "loglogistic") {
        z <- exp(k * (log(t_mid) - eta))
        S_j <- 1 / (1 + z)
      } else if (fam == "lognormal") {
        S_j <- 1 - pnorm((log(t_mid) - eta) / k)
      }
      S_overall <- S_overall * S_j
    }
    
    p_evt <- outcome_params[[event_outcome]]
    k <- p_evt$k; eta <- p_evt$eta; fam <- p_evt$family
    
    if (fam %in% c("weibull", "exponential")) {
      z <- exp(k * (log(t_mid) - eta))
      h_evt <- (k / t_mid) * z
    } else if (fam == "loglogistic") {
      z <- exp(k * (log(t_mid) - eta))
      h_evt <- (k / t_mid) * z / (1 + z)
    } else if (fam == "lognormal") {
      z_score <- (log(t_mid) - eta) / k
      h_evt <- dnorm(z_score) / (t_mid * k * (1 - pnorm(z_score)))
    }
    
    d_cif <- S_overall * h_evt * dt
    cif_values[, i] <- cif_values[, i-1] + d_cif
  }
  
  final_results <- matrix(NA, nrow = n_obs, ncol = length(times))
  colnames(final_results) <- as.character(times)
  
  for (j in seq_along(times)) {
    idx <- match(times[j], sorted_times)
    final_results[, j] <- cif_values[, idx]
  }
  
  as.data.frame(final_results)
}
