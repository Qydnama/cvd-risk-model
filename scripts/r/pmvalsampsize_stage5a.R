args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop(
    paste(
      "Usage: Rscript scripts/r/pmvalsampsize_stage5a.R",
      "OUTPUT.csv PREVALENCE LP_MEAN LP_SD SIMOBS SEED"
    )
  )
}

output_path <- args[[1]]
prevalence <- as.numeric(args[[2]])
lp_mean <- as.numeric(args[[3]])
lp_sd <- as.numeric(args[[4]])
simobs <- as.numeric(args[[5]])
seed <- as.integer(args[[6]])

if (
  any(!is.finite(c(prevalence, lp_mean, lp_sd, simobs, seed))) ||
  prevalence <= 0 || prevalence >= 1 || lp_sd <= 0 || simobs < 1
) {
  stop("Invalid numeric Stage 5a argument")
}

required_version <- "0.1.0"
observed_version <- as.character(utils::packageVersion("pmvalsampsize"))
if (!identical(observed_version, required_version)) {
  stop(
    sprintf(
      "pmvalsampsize version mismatch: observed=%s required=%s",
      observed_version,
      required_version
    )
  )
}

suppressPackageStartupMessages(library(pmvalsampsize))
result <- pmvalsampsize::pmvalsampsize(
  type = "b",
  prevalence = prevalence,
  cstatistic = 0.75,
  cstatciwidth = 0.10,
  cslope = 1,
  csciwidth = 0.40,
  oe = 1,
  oeciwidth = 0.20,
  lpnormal = c(lp_mean, lp_sd),
  simobs = simobs,
  seed = seed,
  graph = FALSE,
  trace = FALSE
)

criteria <- result$results_table
output <- data.frame(
  package = "pmvalsampsize",
  package_version = observed_version,
  type = result$type,
  prevalence = result$prevalence,
  lp_mean = lp_mean,
  lp_sd = lp_sd,
  simobs = simobs,
  seed = seed,
  criterion_oe_n = criteria["Criteria 1 - O/E", "Samp_size"],
  criterion_slope_n = criteria["Criteria 2 - C-slope", "Samp_size"],
  criterion_cstatistic_n = criteria["Criteria 3 - C statistic", "Samp_size"],
  required_n = result$sample_size,
  required_events = result$events,
  achieved_oe_ci_width = result$width_oe,
  achieved_slope_ci_width = result$csciwidth,
  achieved_cstatistic_ci_width = result$cstatciwidth,
  stringsAsFactors = FALSE
)
utils::write.csv(output, output_path, row.names = FALSE, quote = TRUE)
