required_versions <- c(
  survey = "4.5",
  pmsampsize = "1.1.3",
  pmvalsampsize = "0.1.0"
)

cat(sprintf("R=%s\n", R.version.string))
failed <- FALSE
for (package_name in names(required_versions)) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    cat(sprintf("%s=MISSING expected=%s\n", package_name, required_versions[[package_name]]))
    failed <- TRUE
    next
  }
  observed <- as.character(utils::packageVersion(package_name))
  cat(sprintf("%s=%s expected=%s\n", package_name, observed, required_versions[[package_name]]))
  if (!identical(observed, required_versions[[package_name]])) {
    failed <- TRUE
  }
}

if (failed) {
  quit(status = 2)
}
