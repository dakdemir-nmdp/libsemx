#' Extract Random Effects
#'
#' @param object A semx_fit object
#' @param ... Additional arguments (ignored)
#' @return A list of matrices, one per random effect block.
#' @export
semx_ranef <- function(object, ...) {
    if (!inherits(object, "semx_fit")) stop("Object must be a semx_fit object")
    
    res <- list()
    
    # Get raw random effects map
    # It is stored in optimization_result because we added it there in semx.R
    raw_re <- object$optimization_result$random_effects
    if (is.null(raw_re) || length(raw_re) == 0) return(res)
    
    # Iterate over model definitions to structure the output
    model_re <- object$model$random_effects
    
    for (re_def in model_re) {
        name <- re_def$name
        if (is.null(raw_re[[name]])) next
        
        vec <- raw_re[[name]]
        
        # Determine dimensions
        # variables[1] is grouping variable
        # variables[2...] are design variables (intercept, slope, etc.)
        if (length(re_def$variables) < 2) next
        
        group_var <- re_def$variables[[1]]
        design_vars <- re_def$variables[-1]
        n_vars <- length(design_vars)
        
        if (n_vars == 0) next
        
        # Reshape
        # C++ returns [group1_vars, group2_vars, ...]
        # So we fill by row
        mat <- matrix(vec, ncol = n_vars, byrow = TRUE)
        colnames(mat) <- design_vars
        
        # Try to assign row names
        # We need the sorted unique levels of the grouping variable
        if (!is.null(object$factor_levels) && !is.null(object$factor_levels[[group_var]])) {
            levels <- object$factor_levels[[group_var]]
            # Check if length matches
            if (nrow(mat) == length(levels)) {
                rownames(mat) <- levels
            } else {
                # Mismatch can happen if not all levels are present in data
                # Fallback to data values
                if (!is.null(object$data[[group_var]])) {
                    levels <- sort(unique(object$data[[group_var]]))
                    if (nrow(mat) == length(levels)) {
                        rownames(mat) <- as.character(levels)
                    }
                }
            }
        } else {
            # Numeric or character without factor info
            # Use unique values from transformed data (which are 0-based indices)
            if (!is.null(object$data[[group_var]])) {
                levels <- sort(unique(object$data[[group_var]]))
                if (nrow(mat) == length(levels)) {
                    rownames(mat) <- as.character(levels)
                }
            }
        }
        
        res[[name]] <- mat
    }
    
    class(res) <- "semx_ranef"
    res
}

#' @export
print.semx_ranef <- function(x, ...) {
    for (name in names(x)) {
        cat("Random Effect:", name, "\n")
        print(x[[name]])
        cat("\n")
    }
}
