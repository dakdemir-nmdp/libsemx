#' Plot Path Diagram
#'
#' @param fit A semx_fit object.
#' @param ... Additional arguments passed to DiagrammeR::grViz.
#' @return A DiagrammeR object.
#' @keywords internal
semx_plot_path <- function(fit, ...) {
  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
    stop("Package 'DiagrammeR' is required for path diagrams. Please install it.", call. = FALSE)
  }

  model <- fit$model
  vars <- model$variables
  edges <- model$edges
  
  # Variable Kinds
  # observed = 0L, latent = 1L, grouping = 2L, exogenous = 3L
  
  # Edge Kinds
  # loading = 0L, regression = 1L, covariance = 2L
  
  # Build DOT string
  dot <- "digraph semx_path {\n"
  dot <- paste0(dot, "  rankdir=LR;\n")
  dot <- paste0(dot, "  node [fontname=\"Helvetica\"];\n")
  dot <- paste0(dot, "  edge [fontname=\"Helvetica\", fontsize=10];\n")
  
  # Nodes
  for (v_name in names(vars)) {
    v <- vars[[v_name]]
    kind <- v$kind
    
    # Skip internal variables (starting with _)
    if (startsWith(v_name, "_")) next
    
    shape <- "ellipse" # Default to latent
    if (kind == 0L) { # Observed
      shape <- "box"
    } else if (kind == 2L) { # Grouping
      shape <- "hexagon"
    }
    
    label <- v_name
    dot <- paste0(dot, sprintf("  \"%s\" [shape=%s, label=\"%s\"];\n", v_name, shape, label))
  }
  
  # Edges
  for (e in edges) {
    src <- e$source
    tgt <- e$target
    
    # Skip edges involving internal variables
    if (startsWith(src, "_") || startsWith(tgt, "_")) next
    
    param <- e$parameter_id
    kind <- e$kind
    
    # Get parameter estimate if available
    est_label <- ""
    if (!is.null(param) && nzchar(param)) {
       # Find parameter index
       idx <- match(param, fit$parameter_names)
       if (!is.na(idx)) {
         val <- fit$optimization_result$parameters[idx]
         est_label <- sprintf("%.2f", val)
       } else {
         # Try to parse if it's a fixed numeric value
         num_val <- suppressWarnings(as.numeric(param))
         if (!is.na(num_val)) {
             est_label <- sprintf("%.2f", num_val)
         } else {
             est_label <- param
         }
       }
    }
    
    if (kind == 0L || kind == 1L) { # Loading or Regression
      dot <- paste0(dot, sprintf("  \"%s\" -> \"%s\" [label=\"%s\"];\n", src, tgt, est_label))
    } else if (kind == 2L) { # Covariance
      # For variances (self-loops), we might want to display them differently or just as loops
      if (src == tgt) {
          dot <- paste0(dot, sprintf("  \"%s\" -> \"%s\" [dir=both, style=dashed, label=\"%s\", tailport=n, headport=n];\n", src, tgt, est_label))
      } else {
          dot <- paste0(dot, sprintf("  \"%s\" -> \"%s\" [dir=both, style=dashed, label=\"%s\", constraint=false];\n", src, tgt, est_label))
      }
    }
  }
  
  dot <- paste0(dot, "}\n")
  
  DiagrammeR::grViz(dot, ...)
}
