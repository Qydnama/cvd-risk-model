args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript scripts/r/pmvalsampsize_reference.R OUTPUT.txt")
}

suppressPackageStartupMessages(library(pmvalsampsize))
result <- pmvalsampsize::pmvalsampsize(
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
output <- capture.output(dput(result))
writeLines(output, args[[1]], useBytes = TRUE)
