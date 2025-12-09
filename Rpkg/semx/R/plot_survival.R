#' Plot survival curves or CIF
#'
#' @param fit A semx_fit object.
#' @param newdata Optional data frame for prediction. If NULL, uses original data.
#' @param time_grid Optional numeric vector of time points.
#' @param type "survival" or "cif".
#' @param ... Additional arguments passed to plot.
#' @return A ggplot object or list of ggplot objects.
#' @export
semx_plot_survival <- function(fit, newdata = NULL, time_grid = NULL, type = "survival", ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 is required for survival plots")
  }
  
  # Find survival variables
  vars <- fit$model$variables
  surv_vars <- Filter(function(v) v$family %in% c("weibull", "exponential", "loglogistic", "lognormal"), vars)
  
  if (length(surv_vars) == 0) {
    stop("No survival outcomes found in the model")
  }
  
  data <- if (is.null(newdata)) fit$data else newdata
  
  plots <- list()
  
  for (v in surv_vars) {
    var_name <- v$name
    
    # Generate time grid
    if (is.null(time_grid)) {
       if (!is.null(data[[var_name]])) {
         max_t <- max(data[[var_name]], na.rm = TRUE)
       } else {
         max_t <- 100
       }
       times <- seq(0, max_t, length.out = 100)
    } else {
       times <- time_grid
    }
    
    if (type == "survival") {
        surv_probs <- semx_predict_survival(fit, data, times, var_name)
        
        # Reshape for plotting
        plot_df <- data.frame()
        for (i in 1:nrow(surv_probs)) {
            tmp <- data.frame(
                Time = times,
                Survival = as.numeric(surv_probs[i, ]),
                Group = if (nrow(surv_probs) > 1) paste0("Profile ", i) else "Survival"
            )
            plot_df <- rbind(plot_df, tmp)
        }
        
        p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = Time, y = Survival, group = Group, color = Group)) +
          ggplot2::geom_line() +
          ggplot2::ylim(0, 1) +
          ggplot2::labs(title = paste("Survival Curve for", var_name),
                        subtitle = paste("Family:", v$family)) +
          ggplot2::theme_minimal()
          
        if (nrow(surv_probs) > 10) {
          p <- p + ggplot2::theme(legend.position = "none")
        }
        plots[[var_name]] <- p
        
    } else if (type == "cif") {
        # For CIF, we need competing outcomes.
        # Assuming user wants CIF for this outcome against all other survival outcomes?
        # Or user should specify competing outcomes?
        # Let's assume all other survival variables are competing risks.
        competing <- setdiff(names(surv_vars), var_name)
        if (length(competing) == 0) {
            warning(sprintf("No competing outcomes found for %s, skipping CIF", var_name))
            next
        }
        
        cif_probs <- semx_predict_cif(fit, data, times, var_name, competing)
        
        # Reshape
        plot_df <- data.frame()
        for (i in 1:nrow(cif_probs)) {
            tmp <- data.frame(
                Time = times,
                CIF = as.numeric(cif_probs[i, ]),
                Group = if (nrow(cif_probs) > 1) paste0("Profile ", i) else "CIF"
            )
            plot_df <- rbind(plot_df, tmp)
        }
        
        p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = Time, y = CIF, group = Group, color = Group)) +
          ggplot2::geom_line() +
          ggplot2::ylim(0, 1) +
          ggplot2::labs(title = paste("CIF for", var_name),
                        subtitle = paste("Competing:", paste(competing, collapse=", "))) +
          ggplot2::theme_minimal()
          
        if (nrow(cif_probs) > 10) {
          p <- p + ggplot2::theme(legend.position = "none")
        }
        plots[[var_name]] <- p
    }
  }
  
  if (length(plots) == 1) {
    return(plots[[1]])
  } else {
    return(plots)
  }
}
