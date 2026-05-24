# ============================================================================
# 02_run_r_hdclassif.R
#
# HDDC parity check — R side.
#
# Reads the per-dataset artefacts written by ``01_prepare_data.py``:
#   * data/<name>_X.csv           : (n x p) observations, no header.
#   * data/<name>_labels_init.csv : 0-indexed KMeans cluster labels.
#   * data/<name>_meta.json       : n, p, K, cattell_threshold, models, seed.
#
# Fits ``HDclassif::hddc()`` once per (dataset, model) starting from the
# shared init vector, then dumps fitted parameters to ``r_out/`` as CSVs
# so the Python comparison script (``04_compare.py``) can diff them
# field-by-field against the local HDDC fit.
#
# Output files per (dataset, model) pair, written under r_out/:
#   <name>_<model>_bic.csv               : scalar (HDclassif higher-better)
#   <name>_<model>_loglik.csv            : scalar log-likelihood at fixed pt.
#   <name>_<model>_n_parameters.csv      : integer free-param count
#   <name>_<model>_weights.csv           : (K,) mixing proportions
#   <name>_<model>_signal_dims.csv       : (K,) per-cluster d_k
#   <name>_<model>_noise_variances.csv   : (K,) or scalar noise variance(s)
#   <name>_<model>_means.csv             : (K x p) cluster means
#   <name>_<model>_eigenvalues.csv       : (K x p) eigenvalues, padded
#   <name>_<model>_Q_<k>.csv             : (p x d_k) orientation, per cluster
#   <name>_<model>_labels.csv            : (n,) 0-indexed hard labels
#   <name>_<model>_responsibilities.csv  : (n x K) posterior responsibilities
#
# Usage:
#   Rscript hdclassif_parity/02_run_r_hdclassif.R
# ============================================================================

suppressPackageStartupMessages({
  # HDclassif: the reference R implementation of HDDC (Berge et al. 2012).
  # jsonlite: light-weight JSON parser for the meta side-car files.
  if (!requireNamespace("HDclassif", quietly = TRUE)) {
    stop("Install HDclassif first: install.packages('HDclassif')")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Install jsonlite first: install.packages('jsonlite')")
  }
  library(HDclassif)
  library(jsonlite)
})


# ---------------------------------------------------------------------------
# Resolve script directory so paths work no matter the caller's CWD.
# ---------------------------------------------------------------------------
# `Rscript` exposes the script path via `--file=` in commandArgs(); `source()`
# from an interactive R session exposes it via sys.frame(1)$ofile. We try
# both, fall back to the working directory if neither is available.
.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
if (length(.file_arg) > 0) {
  here <- normalizePath(dirname(sub("^--file=", "", .file_arg)))
} else if (!is.null(sys.frames()) && length(sys.frames()) > 0
           && !is.null(sys.frame(1)$ofile)) {
  here <- dirname(sys.frame(1)$ofile)
} else {
  here <- getwd()
}

data_d <- file.path(here, "data")
out_d  <- file.path(here, "r_out")
dir.create(out_d, showWarnings = FALSE, recursive = TRUE)


# ---------------------------------------------------------------------------
# Geometric <-> HDclassif name mapping
# ---------------------------------------------------------------------------
# Our PR exposes HDDC sub-models under three-letter geometric codes
# (e.g. "AVV" for the most general [a_kj b_k Q_k d_k] family); HDclassif
# uses Bouveyron's bracket notation collapsed to a string. This table
# is the single source of truth for the bijection and is referenced
# from docs/HDDC.md §1.
GEOMETRIC_TO_HDCLASSIF <- list(
  AVV = "AkjBkQkDk", AEV = "AkjBQkDk",
  IVV = "AkBkQkDk",  IEV = "AkBQkDk",
  UVV = "ABkQkDk",   UEV = "ABQkDk",
  AVE = "AkjBkQkD",  CVE = "AjBkQkD",
  AEE = "AkjBQkD",   CEE = "AjBQkD",
  IVE = "AkBkQkD",   IEE = "AkBQkD",
  UVE = "ABkQkD",    UEE = "ABQkD"
)


# ---------------------------------------------------------------------------
# Per-(dataset, model) fit + dump
# ---------------------------------------------------------------------------
# resolve_model_spec
#
# @description
#   Parse one element of the meta.json `models` array.
#
#   Returns a list with:
#     * geo        : three-letter geometric code (e.g. "AEE")
#     * hd_name    : HDclassif model string (e.g. "AkjBQkD")
#     * signal_dim : integer if d_k is forced, NA otherwise
#     * tag        : filename tag ("AEE" or "AEE_d2")
#
#   Plain string specs translate to free-d_k (Cattell-driven) fits.
#   Named-list specs of the form list(code="AEE", signal_dim=2L) map
#   to HDclassif's `com_dim` argument for E-suffix models.
resolve_model_spec <- function(spec) {
  if (is.character(spec)) {
    geo <- spec
    signal_dim <- NA_integer_
    tag <- geo
  } else {
    geo <- spec$code
    signal_dim <- as.integer(spec$signal_dim)
    tag <- sprintf("%s_d%d", geo, signal_dim)
  }
  list(
    geo        = geo,
    hd_name    = GEOMETRIC_TO_HDCLASSIF[[geo]],
    signal_dim = signal_dim,
    tag        = tag
  )
}


# fit_and_dump
#
# @description
#   Fit HDclassif::hddc() for one dataset/model pair and persist the
#   fitted parameters as CSVs under r_out/. Failures are caught and
#   reported but do not halt the pipeline (a single failing sub-model
#   should not block parity for the others).
#
# @param name (character) dataset name prefix (e.g. "iris").
# @param spec (resolved model spec from resolve_model_spec()).
# @param X    (matrix, n x p) observations.
# @param init (integer, n)    1-indexed initial cluster labels.
# @param K    (integer)       number of mixture components.
# @param thr  (numeric)       Cattell threshold to forward to hddc().
# @param prefix (character)   path-prefix to which CSV names are appended.
#
# @return invisible NULL. Side effect: writes <prefix><field>.csv files.
fit_and_dump <- function(name, spec, X, init, K, thr, prefix) {
  if (is.null(spec$hd_name)) {
    message(sprintf("skip %s/%s: no HDclassif mapping", name, spec$geo))
    return(invisible(NULL))
  }
  if (is.na(spec$signal_dim)) {
    message(sprintf("R  : %-14s  %-8s  (HDclassif=%s)",
                    name, spec$tag, spec$hd_name))
  } else {
    message(sprintf("R  : %-14s  %-8s  (HDclassif=%s, com_dim=%d)",
                    name, spec$tag, spec$hd_name, spec$signal_dim))
  }

  # Build the hddc() call with com_dim only when d_k is forced;
  # leaving com_dim=NULL keeps the default Cattell-driven behaviour.
  hddc_args <- list(
    data        = X,
    K           = K,
    model       = spec$hd_name,
    init        = "vector",
    init.vector = init,
    threshold   = thr,
    itermax     = 200,
    eps         = 1e-3,
    show        = FALSE
  )
  if (!is.na(spec$signal_dim)) {
    hddc_args$com_dim <- spec$signal_dim
  }

  # tryCatch so one ill-conditioned sub-model does not abort the run.
  fit <- tryCatch(
    do.call(hddc, hddc_args),
    error = function(e) {
      message(sprintf("  FAILED: %s", conditionMessage(e)))
      NULL
    }
  )
  if (is.null(fit)) return(invisible(NULL))

  pK <- ncol(X)

  # --- Scalars (BIC, loglik, n_parameters) ---------------------------------
  # HDclassif's BIC is reported in the higher-is-better convention
  # (2·loglik - ν·log n); the Python comparison script negates it
  # to match sklearn's lower-is-better convention.
  write.table(fit$BIC,
              paste0(prefix, "bic.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(fit$loglik[length(fit$loglik)],
              paste0(prefix, "loglik.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(fit$complexity,
              paste0(prefix, "n_parameters.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")

  # --- Per-cluster scalars (proportions, signal dim, noise variance) -------
  write.table(fit$prop, paste0(prefix, "weights.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(fit$d,    paste0(prefix, "signal_dims.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(fit$b,    paste0(prefix, "noise_variances.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")

  # --- Means (K x p) -------------------------------------------------------
  write.table(fit$mu, paste0(prefix, "means.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")

  # --- Signal eigenvalues padded to (K x p) --------------------------------
  # `fit$a` may be:
  #   - a (K x max_d) matrix when signal eigenvalues are per-cluster
  #     per-axis (the AVV family);
  #   - a length-K vector when isotropic per cluster (e.g. IVV);
  #   - a single scalar when fully tied (e.g. UEE).
  # We always pad to a K x p matrix with zeros in the trailing columns
  # so the Python side can diff a single fixed-shape CSV.
  a_mat <- matrix(0, nrow = K, ncol = pK)
  if (is.matrix(fit$a)) {
    a_mat[, seq_len(ncol(fit$a))] <- fit$a
  } else {
    for (k in seq_len(K)) {
      dk <- fit$d[k]
      a_mat[k, seq_len(dk)] <- if (length(fit$a) == K) fit$a[k] else fit$a
    }
  }
  write.table(a_mat, paste0(prefix, "eigenvalues.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")

  # --- Orientation Q_k per cluster (p x d_k) -------------------------------
  # HDclassif exposes the per-cluster signal subspace bases as a list
  # ``fit$Q`` of (p x d_k) matrices. Confusingly, ``fit$ev`` is the
  # per-cluster eigenvalue *vector*, not the orientation — beware.
  for (k in seq_len(K)) {
    Q_k <- as.matrix(fit$Q[[k]])
    write.table(Q_k,
                paste0(prefix, sprintf("Q_%d.csv", k - 1L)),
                row.names = FALSE, col.names = FALSE, sep = ",")
  }

  # --- Hard labels and posterior responsibilities --------------------------
  # 0-index labels for round-trip with the Python comparison.
  write.table(fit$class - 1L,
              paste0(prefix, "labels.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")
  write.table(fit$posterior,
              paste0(prefix, "responsibilities.csv"),
              row.names = FALSE, col.names = FALSE, sep = ",")

  invisible(NULL)
}


# ---------------------------------------------------------------------------
# Main loop: discover datasets in data/ and dispatch fits per model
# ---------------------------------------------------------------------------
# Dataset discovery is filename-driven (any ``<name>_meta.json`` under
# data/), so adding a dataset to ``01_prepare_data.py`` automatically
# wires it through this script with no edits here.
datasets <- list.files(data_d, pattern = "_meta\\.json$")
datasets <- sub("_meta\\.json$", "", datasets)

for (name in datasets) {
  meta <- fromJSON(file.path(data_d, paste0(name, "_meta.json")))

  # `header = FALSE` because 01_prepare_data.py writes raw CSV without a
  # header row to keep both readers symmetric.
  X <- as.matrix(read.csv(
    file.path(data_d, paste0(name, "_X.csv")),
    header = FALSE
  ))

  # Python wrote 0-indexed labels; R expects 1-indexed cluster IDs.
  init <- as.integer(scan(
    file.path(data_d, paste0(name, "_labels_init.csv")),
    sep = ",", quiet = TRUE
  )) + 1L

  # jsonlite returns the models array either as a character vector
  # (when all entries are strings) or as a list (when any entry is
  # a dict). Normalise to a list so we can iterate uniformly.
  model_specs <- if (is.list(meta$models)) meta$models else as.list(meta$models)

  for (raw_spec in model_specs) {
    spec <- resolve_model_spec(raw_spec)
    prefix <- file.path(out_d, paste0(name, "_", spec$tag, "_"))
    fit_and_dump(
      name   = name,
      spec   = spec,
      X      = X,
      init   = init,
      K      = meta$K,
      thr    = meta$cattell_threshold,
      prefix = prefix
    )
  }
}

message("R parity dump complete -> ", out_d)
