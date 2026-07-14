args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript scripts/r/survey_reference.R INPUT.csv OUTPUT.csv")
}

suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")
options(survey.adjust.domain.lonely = TRUE)

input <- utils::read.csv(args[[1]], stringsAsFactors = FALSE)
required <- c("outcome", "weight", "strata", "psu")
missing <- setdiff(required, names(input))
if (length(missing) > 0) {
  stop(sprintf("Missing fixture columns: %s", paste(missing, collapse = ", ")))
}

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
design_effect <- as.numeric(survey::deff(estimate))
kish_ess <- sum(input$weight)^2 / sum(input$weight^2)

output <- data.frame(
  p_hat = as.numeric(stats::coef(estimate)),
  se = as.numeric(survey::SE(estimate)),
  ci_low = as.numeric(ci[1]),
  ci_high = as.numeric(ci[2]),
  deff = design_effect,
  design_df = survey::degf(design),
  kish_ess = kish_ess
)
utils::write.csv(output, args[[2]], row.names = FALSE, quote = FALSE)
