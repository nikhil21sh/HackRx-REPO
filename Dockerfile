# Use an official, lightweight Python image.
FROM python:3.10-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file first for Docker layer caching.
COPY requirements.txt requirements.txt

# Install the Python dependencies.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of your application code into the container.
COPY . .

# --- Start Command (FIXED) ---
# Define the command to run when the container starts.
# We add "--timeout 300" to give the worker 300 seconds (5 minutes)
# to load the heavy AI models during startup, preventing a timeout.
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "300"]
