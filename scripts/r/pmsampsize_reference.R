args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop(
    "Usage: Rscript scripts/r/pmsampsize_reference.R prevalence csrsquared parameters shrinkage OUTPUT.txt"
  )
}

suppressPackageStartupMessages(library(pmsampsize))
result <- pmsampsize::pmsampsize(
  type = "b",
  prevalence = as.numeric(args[[1]]),
  csrsquared = as.numeric(args[[2]]),
  parameters = as.integer(args[[3]]),
  shrinkage = as.numeric(args[[4]])
)
output <- capture.output(dput(result))
writeLines(output, args[[5]], useBytes = TRUE)
