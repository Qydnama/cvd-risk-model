args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript scripts/r/stage6_design_risk_precision.R INPUT.csv OUTPUT.csv")
}

required_version <- "4.5"
observed_version <- as.character(utils::packageVersion("survey"))
if (!identical(observed_version, required_version)) {
  stop(sprintf("survey version mismatch: observed=%s required=%s", observed_version, required_version))
}

suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")
options(survey.adjust.domain.lonely = TRUE)

input <- utils::read.csv(args[[1]], stringsAsFactors = FALSE)
required <- c("cohort", "outcome", "weight", "strata", "psu")
missing <- setdiff(required, names(input))
if (length(missing) > 0) {
  stop(sprintf("Missing design-risk fields: %s", paste(missing, collapse = ", ")))
}

rows <- list()
for (cohort_name in unique(input$cohort)) {
  frame <- input[input$cohort == cohort_name, , drop = FALSE]
  if (anyNA(frame[required]) || any(frame$weight <= 0)) {
    stop(sprintf("%s has missing design values or non-positive weights", cohort_name))
  }
  if (!all(frame$outcome %in% c(0, 1))) {
    stop(sprintf("%s outcome is not binary", cohort_name))
  }
  design <- survey::svydesign(
    ids = ~psu,
    strata = ~strata,
    weights = ~weight,
    nest = TRUE,
    data = frame
  )
  estimate <- survey::svyciprop(
    ~I(outcome == 1),
    design,
    method = "logit",
    level = 0.95
  )
  interval <- stats::confint(estimate)
  point <- as.numeric(stats::coef(estimate))
  full_width <- as.numeric(interval[2] - interval[1])
  rows[[length(rows) + 1]] <- data.frame(
    cohort = cohort_name,
    n_nominal = nrow(frame),
    raw_events = sum(frame$outcome),
    weighted_prevalence = point,
    ci_lower = as.numeric(interval[1]),
    ci_upper = as.numeric(interval[2]),
    full_ci_width = full_width,
    target_full_ci_width = 0.005,
    passes_target = full_width <= 0.005,
    design_df = survey::degf(design),
    method = "survey_svyciprop_logit",
    stringsAsFactors = FALSE
  )
}

utils::write.csv(do.call(rbind, rows), args[[2]], row.names = FALSE, quote = FALSE)
