FROM python:3.13-slim
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency definitions
COPY pyproject.toml uv.lock* ./

# Install dependencies (production only)
RUN uv pip install --system -r pyproject.toml

# Copy source code
COPY . /app

# Expose the port
EXPOSE 8080

# Start FastAPI with Uvicorn
CMD ["python","main.py"]