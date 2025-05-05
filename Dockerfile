# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim AS base

# 2. Set environment variables
# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
# Set default backend URL (MUST be overridden at runtime for different instances)
ENV BACKEND_API_URL="http://localhost:8001/v1"
# Set default log file path inside the container (can be overridden)
ENV LOG_FILE_PATH="/app/logs/app.log"
# Set default logging verbosity (can be overridden)
ENV VERBOSE_LOGGING="true"
# Set default max retries (can be overridden)
ENV DEFAULT_MAX_RETRIES=1
# Set default API key (can be overridden)
ENV BACKEND_API_KEY="sk-1111"

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install system dependencies if needed (e.g., for pygraphviz)
# Uncomment the next line if you need pygraphviz and include it in requirements.txt
# RUN apt-get update && apt-get install -y --no-install-recommends graphviz && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the application code into the container
COPY main_langgraph_enhanced.py .

# 7. Create the default log directory
# Ensure the directory for the default LOG_FILE_PATH exists
# Use RUN to execute this command during the build process
RUN mkdir -p $(dirname ${LOG_FILE_PATH})
# Optional: If running as non-root user later, ensure permissions.
# For now, running as root (default) is fine.
# RUN chown <user>:<group> $(dirname ${LOG_FILE_PATH})

# 8. Expose the port the app runs on
# This informs Docker that the container listens on this port.
# The actual mapping happens during `docker run`.
EXPOSE 8000

# 9. Define the command to run the application
# Use the PORT environment variable.
# DO NOT use --reload in the container's run command.
CMD ["uvicorn", "main_langgraph_enhanced:app", "--host", "0.0.0.0", "--port", "8000"]