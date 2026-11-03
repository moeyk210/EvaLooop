# Sandbox image for running EvaLoop experiments.
# Model-generated code is executed by the evaluator, so everything runs
# inside this container as an unprivileged user with no host files mounted
# except the repo (read-only) and a results directory.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only torch: models are called through the API, so no CUDA is needed
# and the image stays ~1.5 GB smaller.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"

WORKDIR /opt/evaloop
COPY requirements.txt pyproject.toml README.md MANIFEST.in ./
RUN pip install -r requirements.txt pytest

COPY evaloop ./evaloop
COPY tests ./tests
RUN pip install --no-deps -e .

# Unprivileged user; results and the EvalPlus dataset cache live in /work.
RUN useradd --create-home --uid 1000 runner \
    && mkdir -p /work/results /work/cache \
    && chown -R runner:runner /work
USER runner
ENV XDG_CACHE_HOME=/work/cache \
    HF_HOME=/work/cache/huggingface \
    MPLCONFIGDIR=/tmp/mpl
WORKDIR /work

ENTRYPOINT ["evaloop"]
