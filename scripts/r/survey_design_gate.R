args <- commandArgs(trailingOnly = TRUE)
`%+%` <- function(left, right) paste0(left, right)
if (length(args) != 7) {
  stop(
    "Usage: Rscript scripts/r/survey_design_gate.R INPUT.csv OUTPUT.csv " %+%
      "R2_FRACTION PARAMETERS SHRINKAGE MAX_CI_HALF_WIDTH MIN_DESIGN_DF"
  )
}

suppressPackageStartupMessages(library(survey))
suppressPackageStartupMessages(library(pmsampsize))
options(survey.lonely.psu = "adjust")
options(survey.adjust.domain.lonely = TRUE)

input <- utils::read.csv(args[[1]], stringsAsFactors = FALSE)
required <- c("cohort", "outcome", "weight", "strata", "psu")
missing <- setdiff(required, names(input))
if (length(missing) > 0) {
  stop(sprintf("Missing design columns: %s", paste(missing, collapse = ", ")))
}
r2_fraction <- as.numeric(args[[3]])
parameters <- as.integer(args[[4]])
shrinkage <- as.numeric(args[[5]])
max_ci_half_width <- as.numeric(args[[6]])
minimum_design_df <- as.integer(args[[7]])

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
  estimate <- survey::svymean(~outcome, design)
  interval <- survey::svyciprop(~I(outcome == 1), design, method = "logit", level = 0.95)
  ci <- stats::confint(interval)
  p_hat <- as.numeric(stats::coef(estimate))
  variance <- as.numeric(stats::vcov(estimate))
  n_nominal <- nrow(frame)
  srs_variance <- p_hat * (1 - p_hat) / n_nominal
  if (!is.finite(variance) || variance < 0 || !is.finite(srs_variance) || srs_variance <= 0) {
    stop(sprintf("%s has invalid design or SRS variance", cohort_name))
  }
  deff <- variance / srs_variance
  deff_gate <- max(1, deff)
  kish_ess <- sum(frame$weight)^2 / sum(frame$weight^2)
  design_ess <- n_nominal / deff_gate
  half_width <- max(p_hat - ci[1], ci[2] - p_hat)

  pms_required <- NA_integer_
  expected_r2 <- NA_real_
  required_nominal <- NA_integer_
  pass_information <- NA
  if (cohort_name == "nhis_development") {
    r2_max <- 1 - exp(2 * (p_hat * log(p_hat) + (1 - p_hat) * log(1 - p_hat)))
    expected_r2 <- r2_fraction * r2_max
    pms <- pmsampsize::pmsampsize(
      type = "b",
      prevalence = p_hat,
      csrsquared = expected_r2,
      parameters = parameters,
      shrinkage = shrinkage
    )
    pms_required <- as.integer(pms$sample_size)
    required_nominal <- as.integer(ceiling(pms_required * deff_gate))
    pass_information <- design_ess >= pms_required
  }

  rows[[length(rows) + 1]] <- data.frame(
    cohort = cohort_name,
    n_nominal = n_nominal,
    raw_events = sum(frame$outcome),
    p_hat = p_hat,
    se = sqrt(variance),
    ci_low = as.numeric(ci[1]),
    ci_high = as.numeric(ci[2]),
    max_ci_half_width = half_width,
    deff_y = deff,
    deff_y_gate = deff_gate,
    n_eff_design = design_ess,
    n_eff_kish = kish_ess,
    design_df = survey::degf(design),
    expected_r2_cs = expected_r2,
    pmsampsize_required_n = pms_required,
    required_n_nominal = required_nominal,
    pass_information = pass_information,
    pass_risk_precision = half_width <= max_ci_half_width,
    pass_design_df = survey::degf(design) >= minimum_design_df
  )
}

output <- do.call(rbind, rows)
utils::write.csv(output, args[[2]], row.names = FALSE, quote = FALSE, na = "")
