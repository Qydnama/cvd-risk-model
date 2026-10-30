# Assignment 5 DevOps and CI/CD

This exercise automates the existing `minimal_v1` CVD research model. It does not
train another model, create a clinical service, or access participant data.
GitHub Actions replaces Jenkins with the instructor's approval. Kubernetes is
discussed in the report but is not required for this local batch deployment.

## Files

- `Dockerfile`: digest-pinned Python/uv base images, locked dependencies,
  multi-stage build and unprivileged runtime user.
- `.dockerignore`: allowlist excluding datasets, Git history, credentials,
  virtual environments and unrelated research outputs.
- `src/cvd_risk/devops_demo.py`: checksum and runtime verification followed by
  calibrated inference on complete synthetic records.
- `devops/synthetic_inputs.json`: three fictional demonstration records.
- `.github/workflows/assignment5.yml`: tests, image build, verification,
  publication to GHCR, and local deployment.
- `devops/Deploy-ResearchDemo.ps1`: installs the exact image digest, verifies it
  before promotion, and keeps the previous deployment manifest.

## Local checks

With Python 3.12 and uv available:

```powershell
uv sync --locked --all-extras --dev
uv run --frozen pytest -q -m "not requires_local_research_artifacts"
uv run --frozen python -m cvd_risk.devops_demo
```

With Docker Desktop running in Linux-container mode:

```powershell
$revision = git rev-parse HEAD
docker build --build-arg "GIT_COMMIT=$revision" -t cvd-research:local .
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=16m cvd-research:local
```

The output must report `PASS`, four verified binary artifacts, three finite
probabilities, `refit_performed: false` and `clinical_deployment_allowed: false`.
These are software checks, not new evidence of predictive accuracy or clinical
validity. The full Git-based Stage 9 checks run in the CI test environment;
the runtime container checks its installed files and versions without Git.

## Automatic workflow

Pushes to `codex/assignment-5-devops` and, after integration, `main` run all
three jobs. Pull requests run cloud-hosted tests and container verification;
they do not publish or deploy. A failed test blocks the image job, and an
unsuccessful image check blocks deployment.

The image is published as
`ghcr.io/qydnama/cvd-risk-model-research:sha-<full-commit>` using the job's
temporary `GITHUB_TOKEN`. CD downloads the immutable `@sha256:...` reference
returned by that build; no personal registry password is stored in the repo.

## Local CD target

The destination is `%LOCALAPPDATA%\CvdResearchDemo`, outside the Git checkout
and the runner workspace. It contains `current.json`, versioned deployment
and verification records, a previous manifest after updates, and a reusable
`Run-CvdDemo.ps1` launcher. For this batch tool, deployment installs a version
that remains runnable after the Actions job ends; no HTTP server is needed.

1. Keep Docker Desktop running and authenticate GitHub CLI as the repo owner.
2. On a trusted checkout, run the following command to start a one-job runner:

   ```powershell
   powershell -ExecutionPolicy Bypass -File devops/Start-Assignment5Runner.ps1
   ```

3. Push an update to the assignment branch. CI runs in GitHub; the successful
   build triggers CD on the local runner. If the runner is offline, the CD job
   waits until the runner starts.
4. After successful deployment, independently run the installed version:

   ```powershell
   powershell -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\CvdResearchDemo\Run-CvdDemo.ps1"
   ```

Because this is a public repository, the Windows runner is **ephemeral**: it
executes one job and unregisters. Start it only for a trusted assignment push;
do not attach a permanently running personal-computer runner to arbitrary
pull requests. Restart the setup script for a subsequent demonstration.
No Windows service or scheduled task is installed.

The deployment script checks the candidate before replacing `current.json`.
A failed pull, inference check, revision check or artifact verification leaves
the active deployment unchanged. The old image is retained for recovery.

## Submission evidence

Submit the English report and screenshots of the actual Actions graph,
test/build logs and successful CD. Include the repository branch, workflow,
successful run and commit links. Also show the installed launcher running
after the ephemeral runner has exited. This demonstrates persistent deployment
rather than only a temporary container smoke test.
