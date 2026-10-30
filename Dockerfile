# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.3@sha256:90bbb3c16635e9627f49eec6539f956d70746c409209041800a0280b93152823 AS uv
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# The selected estimator is logistic regression; exclude the unused XGBoost/GPU stack.
RUN uv export --quiet --frozen --extra modeling --no-dev --no-emit-project \
      --no-emit-package xgboost --no-emit-package nvidia-nccl-cu12 \
      --output-file /tmp/runtime-requirements.txt \
    && uv venv .venv \
    && uv pip sync --python .venv/bin/python /tmp/runtime-requirements.txt
COPY src ./src
RUN uv pip install --python .venv/bin/python --no-deps .

FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS runtime
ARG GIT_COMMIT=local-checkout
LABEL org.opencontainers.image.source="https://github.com/Qydnama/cvd-risk-model" \
      org.opencontainers.image.title="CVD research demo" \
      org.opencontainers.image.revision=$GIT_COMMIT
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CVD_SOURCE_REVISION=$GIT_COMMIT MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY uv.lock ./
COPY artifacts/minimal_v1/*.joblib ./artifacts/minimal_v1/
COPY artifacts/minimal_v1_research_release/input_schema.json artifacts/minimal_v1_research_release/release_manifest.json ./artifacts/minimal_v1_research_release/
COPY devops/synthetic_inputs.json ./devops/synthetic_inputs.json
USER 10001:10001
ENTRYPOINT ["python", "-m", "cvd_risk.devops_demo"]
