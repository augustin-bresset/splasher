# Splasher demo — container image (also used by the Hugging Face Space, Docker SDK).
#
# Serves the FastAPI backend + web front on the synthetic `demo` source. Only the
# `api` extra is installed (FastAPI + uvicorn + numpy) — no desktop/Qt deps, so the
# image stays small. Listens on 0.0.0.0:7860 (the port Hugging Face Spaces expects).
#
#   docker build -t splasher .
#   docker run --rm -p 7860:7860 splasher   # -> http://localhost:7860
FROM python:3.12-slim

# uv (pinned) for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /uvx /bin/

# Hugging Face Spaces runs containers as uid 1000; create a matching non-root user
# so the venv and caches live in a writable home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Resolve and install dependencies first (cached unless pyproject/uv.lock change),
# without the project itself. README/LICENSE are referenced by pyproject metadata.
COPY --chown=user pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --extra api --frozen --no-dev --no-install-project

# Then the source, and install the project (web assets are packaged under splasher/web).
COPY --chown=user splasher ./splasher
RUN uv sync --extra api --frozen --no-dev

EXPOSE 7860
CMD ["/app/.venv/bin/splasher", "demo", "--serve", "--host", "0.0.0.0", "--port", "7860"]
