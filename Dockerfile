FROM python:3.11.12-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

# Install the package via uv
RUN uv pip install --system .

# Run as non-root user
RUN useradd --create-home agent
USER agent

# Default command: run the CLI
ENTRYPOINT ["research-agent"]
CMD ["--help"]
