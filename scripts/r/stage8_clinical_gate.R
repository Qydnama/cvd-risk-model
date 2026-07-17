args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) {
  stop(
    paste0(
      "Usage: Rscript scripts/r/stage8_clinical_gate.R INPUT.csv OUTPUT.csv ",
      "R2_FRACTION PARAMETERS SHRINKAGE MAX_CI_HALF_WIDTH MIN_DESIGN_DF"
    )
  )
}

suppressPackageStartupMessages(library(survey))
suppressPackageStartupMessages(library(pmsampsize))
options(survey.lonely.psu = "adjust")
options(survey.adjust.domain.lonely = TRUE)

input <- utils::read.csv(args[[1]], stringsAsFactors = FALSE)
required <- c("outcome", "weight", "strata", "psu")
missing <- setdiff(required, names(input))
if (length(missing) > 0) {
  stop(sprintf("Missing design columns: %s", paste(missing, collapse = ", ")))
}
if (anyNA(input[required]) || any(input$weight <= 0)) {
  stop("Clinical gate input has missing design values or non-positive weights")
}
if (!all(input$outcome %in% c(0, 1))) {
  stop("Clinical gate outcome is not binary")
}

r2_fraction <- as.numeric(args[[3]])
parameters <- as.integer(args[[4]])
shrinkage <- as.numeric(args[[5]])
max_ci_half_width <- as.numeric(args[[6]])
minimum_design_df <- as.integer(args[[7]])

design <- survey::svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~weight,
  nest = TRUE,
  data = input
)
estimate <- survey::svymean(~outcome, design)
interval <- survey::svyciprop(~I(outcome == 1), design, method = "logit", level = 0.95)
ci <- stats::confint(interval)
p_hat <- as.numeric(stats::coef(estimate))
variance <- as.numeric(stats::vcov(estimate))
n_nominal <- nrow(input)
srs_variance <- p_hat * (1 - p_hat) / n_nominal
if (!is.finite(variance) || variance < 0 || !is.finite(srs_variance) || srs_variance <= 0) {
  stop("Clinical gate has invalid design or SRS variance")
}
deff <- variance / srs_variance
deff_gate <- max(1, deff)
n_eff_design <- n_nominal / deff_gate
n_eff_kish <- sum(input$weight)^2 / sum(input$weight^2)
design_df <- survey::degf(design)
half_width <- max(p_hat - ci[1], ci[2] - p_hat)

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

output <- data.frame(
  cohort = "nhanes_internal_clinical_core",
  n_nominal = n_nominal,
  raw_events = sum(input$outcome),
  p_hat = p_hat,
  se = sqrt(variance),
  ci_low = as.numeric(ci[1]),
  ci_high = as.numeric(ci[2]),
  max_ci_half_width = half_width,
  deff_y = deff,
  deff_y_gate = deff_gate,
  n_eff_design = n_eff_design,
  n_eff_kish = n_eff_kish,
  design_df = design_df,
  expected_r2_cs = expected_r2,
  pmsampsize_required_n = pms_required,
  required_n_nominal = as.integer(ceiling(pms_required * deff_gate)),
  pass_information = n_eff_design >= pms_required,
  pass_risk_precision = half_width <= max_ci_half_width,
  pass_design_df = design_df >= minimum_design_df,
  survey_version = as.character(utils::packageVersion("survey")),
  pmsampsize_version = as.character(utils::packageVersion("pmsampsize"))
)
utils::write.csv(output, args[[2]], row.names = FALSE, quote = FALSE)

