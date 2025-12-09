#' Plot Reaction Norms (GxE)
#'
#' Visualizes the interaction between genotype and environment by plotting
#' reaction norms (phenotype vs environment) for each genotype.
#'
#' @param object A semx_fit object.
#' @param x The name of the environmental variable (x-axis).
#' @param group The name of the grouping variable (genotype/individual).
#' @param response The name of the response variable.
#' @param add_points Logical. If TRUE, adds observed data points.
#' @param ... Additional arguments passed to plot.
#' @export
plot_reaction_norms <- function(object, x, group, response, add_points = TRUE, ...) {
    data <- object$data
    if (is.null(data)) stop("Original data not found in semx_fit object.")
    
    if (is.list(data) && !is.data.frame(data)) {
        # Multi-group data?
        # For now, assume single group data or merged
        # If list of dataframes, maybe bind them?
        if (is.data.frame(data[[1]])) {
            data <- do.call(rbind, data)
        } else {
             stop("Data format not supported for plotting.")
        }
    }
    
    x_col <- data[[x]]
    is_categorical <- is.factor(x_col) || is.character(x_col)
    
    params <- object$fit_result$optimization_result$parameters
    names(params) <- object$fit_result$parameter_names
    
    ranefs <- semx_ranef(object)
    re_block <- ranefs[[group]]
    if (is.null(re_block)) {
        stop(sprintf("Random effects for group '%s' not found.", group))
    }
    
    if (is_categorical) {
        # Categorical X (Interaction Plot)
        levels <- sort(unique(x_col))
        n_levels <- length(levels)
        
        # Calculate predicted values for each group x level
        # Fixed effects
        # Assuming standard contrast coding (Treatment/Dummy)
        # Intercept = Level 1
        # Level k = Intercept + beta_xLevelk
        
        beta_int <- params[paste0("beta_", response, "_on__intercept")]
        if (is.na(beta_int)) beta_int <- 0
        
        fixed_means <- numeric(n_levels)
        names(fixed_means) <- levels
        
        fixed_means[1] <- beta_int
        
        for (k in 2:n_levels) {
            lvl <- levels[k]
            # Parameter name might be "beta_y_on_xLevel"
            # Need to match how semx names dummy variables
            # Usually: beta_{response}_on_{x}{Level}
            # But semx parser might name them differently?
            # Let's assume standard naming for now.
            param_name <- paste0("beta_", response, "_on_", x, lvl)
            val <- params[param_name]
            if (is.na(val)) val <- 0
            fixed_means[k] <- beta_int + val
        }
        
        # Random effects
        # Case 1: (0 + x | group) -> Random effect for each level
        # Case 2: (1 | group) + (1 | group:x) -> Random intercept + Random interaction
        
        # We check columns of re_block
        re_cols <- colnames(re_block)
        
        # Prepare matrix of predictions: Rows = Groups, Cols = Levels
        groups <- rownames(re_block)
        preds <- matrix(NA, nrow = length(groups), ncol = n_levels)
        rownames(preds) <- groups
        colnames(preds) <- levels
        
        for (i in seq_along(groups)) {
            for (k in seq_along(levels)) {
                lvl <- levels[k]
                
                # Start with fixed mean
                val <- fixed_means[k]
                
                # Add random effects
                # 1. Random Intercept?
                if ("_intercept" %in% re_cols) {
                    val <- val + re_block[i, "_intercept"]
                }
                
                # 2. Random Slope/Level?
                # If column matches level name exactly?
                # Or "xLevel"?
                
                # Try exact match
                if (lvl %in% re_cols) {
                    val <- val + re_block[i, lvl]
                } else {
                    # Try with variable name prefix
                    col_name <- paste0(x, lvl)
                    if (col_name %in% re_cols) {
                        val <- val + re_block[i, col_name]
                    }
                }
                
                preds[i, k] <- val
            }
        }
        
        # Plot
        matplot(seq_along(levels), t(preds), type = "l", lty = 1, col = rgb(0, 0, 1, 0.3),
                xaxt = "n", xlab = x, ylab = response, ...)
        axis(1, at = seq_along(levels), labels = levels)
        
        # Add population mean
        lines(seq_along(levels), fixed_means, col = "red", lwd = 3)
        
        if (add_points) {
            # Map x to indices
            x_idx <- as.numeric(factor(x_col, levels = levels))
            points(jitter(x_idx), data[[response]], col = "gray", pch = 1, cex = 0.5)
        }
        
    } else {
        # Continuous X (Reaction Norms)
        beta_int <- params[paste0("beta_", response, "_on__intercept")]
        if (is.na(beta_int)) beta_int <- 0
        
        beta_slope <- params[paste0("beta_", response, "_on_", x)]
        if (is.na(beta_slope)) beta_slope <- 0
        
        u_int <- if ("_intercept" %in% colnames(re_block)) re_block[, "_intercept"] else rep(0, nrow(re_block))
        u_slope <- if (x %in% colnames(re_block)) re_block[, x] else rep(0, nrow(re_block))
        
        x_range <- range(x_col, na.rm = TRUE)
        x_grid <- seq(x_range[1], x_range[2], length.out = 100)
        
        plot(x_col, data[[response]], type = "n", xlab = x, ylab = response, ...)
        
        if (add_points) {
            points(x_col, data[[response]], col = "gray", pch = 1)
        }
        
        groups <- rownames(re_block)
        for (i in seq_along(groups)) {
            int_i <- beta_int + u_int[i]
            slope_i <- beta_slope + u_slope[i]
            y_grid <- int_i + slope_i * x_grid
            lines(x_grid, y_grid, col = rgb(0, 0, 1, 0.3))
        }
        
        lines(x_grid, beta_int + beta_slope * x_grid, col = "red", lwd = 3)
    }
}
