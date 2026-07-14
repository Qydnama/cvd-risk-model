suppressPackageStartupMessages(library(survey))
suppressPackageStartupMessages(library(pmsampsize))
suppressPackageStartupMessages(library(pmvalsampsize))

options(survey.lonely.psu = "adjust")
options(survey.adjust.domain.lonely = TRUE)

input <- utils::read.csv("tests/fixtures/r/survey_binary.csv", stringsAsFactors = FALSE)
expected <- utils::read.csv(
  "tests/fixtures/r/survey_binary_expected.csv",
  stringsAsFactors = FALSE
)
design <- survey::svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~weight,
  nest = TRUE,
  data = input
)
estimate <- survey::svymean(~outcome, design, deff = TRUE)
interval <- survey::svyciprop(~I(outcome == 1), design, method = "logit", level = 0.95)
ci <- stats::confint(interval)
observed <- data.frame(
  p_hat = as.numeric(stats::coef(estimate)),
  se = as.numeric(survey::SE(estimate)),
  ci_low = as.numeric(ci[1]),
  ci_high = as.numeric(ci[2]),
  deff = as.numeric(survey::deff(estimate)),
  design_df = survey::degf(design),
  kish_ess = sum(input$weight)^2 / sum(input$weight^2)
)

absolute_columns <- c("p_hat", "se", "ci_low", "ci_high")
absolute_error <- max(abs(observed[absolute_columns] - expected[absolute_columns]))
if (!is.finite(absolute_error) || absolute_error > 1e-10) {
  stop(sprintf("survey absolute tolerance failed: %.17g", absolute_error))
}
relative_columns <- c("deff", "kish_ess")
relative_error <- max(
  abs(observed[relative_columns] - expected[relative_columns]) /
    abs(expected[relative_columns])
)
if (!is.finite(relative_error) || relative_error > 1e-8) {
  stop(sprintf("survey relative tolerance failed: %.17g", relative_error))
}
if (!identical(as.integer(observed$design_df), as.integer(expected$design_df))) {
  stop("survey design degrees of freedom differ from the frozen reference")
}

sample_expected <- utils::read.csv(
  "tests/fixtures/r/sample_size_expected.csv",
  stringsAsFactors = FALSE
)
pms <- pmsampsize::pmsampsize(
  type = "b",
  prevalence = 0.02,
  csrsquared = 0.10,
  parameters = 15,
  shrinkage = 0.90
)
pmval <- pmvalsampsize::pmvalsampsize(
  type = "b",
  prevalence = 0.02,
  cstatistic = 0.75,
  cstatciwidth = 0.10,
  cslope = 1,
  csciwidth = 0.40,
  oe = 1,
  oeciwidth = 0.20,
  lpnormal = c(-3.0, 1.0),
  simobs = 1000000,
  seed = 20260714,
  graph = FALSE,
  trace = FALSE
)
observed_n <- c(pmsampsize = pms$sample_size, pmvalsampsize = pmval$sample_size)
expected_n <- stats::setNames(sample_expected$required_n, sample_expected$package)
if (!identical(as.integer(observed_n), as.integer(expected_n[names(observed_n)]))) {
  stop("sample-size output differs from the frozen integer references")
}

cat(sprintf("survey_max_absolute_error=%.17g\n", absolute_error))
cat(sprintf("survey_max_relative_error=%.17g\n", relative_error))
cat(sprintf("pmsampsize_required_n=%d\n", as.integer(observed_n[["pmsampsize"]])))
cat(sprintf("pmvalsampsize_required_n=%d\n", as.integer(observed_n[["pmvalsampsize"]])))
cat("R_REFERENCE_TESTS=PASS\n")
