args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript scripts/r/label_free_design_effects.R INPUT.csv OUTPUT.csv")
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
required <- c("cohort", "weight", "strata", "psu", "linear_predictor", "prediction")
missing <- setdiff(required, names(input))
if (length(missing) > 0) {
  stop(sprintf("Missing label-free design columns: %s", paste(missing, collapse = ", ")))
}

rows <- list()
for (cohort_name in unique(input$cohort)) {
  frame <- input[input$cohort == cohort_name, , drop = FALSE]
  if (anyNA(frame[required]) || any(frame$weight <= 0)) {
    stop(sprintf("%s has missing design values or non-positive weights", cohort_name))
  }
  design <- survey::svydesign(
    ids = ~psu,
    strata = ~strata,
    weights = ~weight,
    nest = TRUE,
    data = frame
  )
  for (variable_name in c("linear_predictor", "prediction")) {
    values <- frame[[variable_name]]
    estimate <- survey::svymean(stats::as.formula(paste0("~", variable_name)), design)
    weighted_mean <- stats::weighted.mean(values, frame$weight)
    population_variance <- stats::weighted.mean((values - weighted_mean)^2, frame$weight)
    srs_variance <- population_variance / nrow(frame)
    design_variance <- as.numeric(stats::vcov(estimate))
    if (
      !is.finite(design_variance) || design_variance < 0 ||
      !is.finite(srs_variance) || srs_variance <= 0
    ) {
      stop(sprintf("%s/%s has invalid design or SRS variance", cohort_name, variable_name))
    }
    rows[[length(rows) + 1]] <- data.frame(
      cohort = cohort_name,
      variable = variable_name,
      n_nominal = nrow(frame),
      weighted_mean = weighted_mean,
      weighted_population_sd = sqrt(population_variance),
      design_se = sqrt(design_variance),
      srs_se = sqrt(srs_variance),
      design_effect = design_variance / srs_variance,
      design_df = survey::degf(design),
      stringsAsFactors = FALSE
    )
  }
}

utils::write.csv(do.call(rbind, rows), args[[2]], row.names = FALSE, quote = FALSE)
