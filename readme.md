# FastAgentAPI (LangGraph + FastAPI Wrapper)

## Overview

This project provides a FastAPI wrapper, **FastAgentAPI**, that acts as an OpenAI-compatible endpoint (`/v1/chat/completions`). It leverages LangGraph to manage complex conversational flows or agentic behaviors with a backend Large Language Model (LLM). Key features include:

* **LangGraph Integration:** Orchestrates calls to the backend LLM, including state management, conditional logic, and retries. Suitable for building robust LLM agents.
* **Structured JSON Output:** Supports OpenAI's `response_format={ "type": "json_object" }` including schema validation.
* **Tool Call Handling:** Can interpret and format tool call requests/responses in text mode.
* **Think Tag Stripping:** Automatically removes `<think>...</think>` tags from LLM responses when generating structured JSON.
* **Robust Logging:** Implements Python's standard `logging` module, outputting to both console (for `docker logs`) and a configurable log file.
* **Dockerized:** Includes a Dockerfile for easy building and deployment, allowing multiple instances with different configurations.
* **Configurable:** Uses environment variables for essential settings like backend URL, API keys, logging paths, and verbosity.

## Features

* OpenAI Compatible `/v1/chat/completions` endpoint.
* Handles `response_format: json_object` with schema validation.
* Handles standard text generation and OpenAI-like tool calls.
* Configurable retries for backend LLM calls.
* Detailed logging to console and file.
* Optional graph visualization output (`flow.png`).
* Ready for containerization with Docker.

## Prerequisites

### Local Development

* Python 3.11+
* Pip (Python package installer)
* Access to a backend LLM API service.
* (Optional) Graphviz system library (`sudo apt-get install graphviz` or `brew install graphviz`) if you want `pygraphviz` to generate the `flow.png` graph image locally.

### Docker Deployment

* Docker Engine installed and running.

## Setup and Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Teachings/FastAgentAPI.git
    cd FastAgentAPI
    ```

2.  **Create and activate a virtual environment:**
    * Using Conda:
        ```bash
        conda create -n fastagentapi python=3.11 -y
        conda activate fastagentapi
        ```
    * Using venv:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `pygraphviz` installation might require the Graphviz system library first. See Optional Visualization section).*

4.  **Set Environment Variables:**
    Create a `.env` file in the project root or export the following variables in your terminal. Ensure you have added `.env` to your `.gitignore` file.
    ```dotenv
    # Example .env file
    BACKEND_API_URL="<your_backend_llm_api_url>/v1" # e.g., http://192.168.1.11:8001/v1
    BACKEND_API_KEY="<your_backend_api_key>"      # Optional, if your backend needs it
    LOCAL_PORT="8002"                             # Port for local Uvicorn server
    LOG_FILE_PATH="./logs/app_local.log"          # Path for the local log file
    VERBOSE_LOGGING="true"                        # Set to "false" for less detailed logs
    # DEFAULT_MAX_RETRIES="1"                     # Optional: Default retry attempts
    ```

5.  **Run the application locally:**
    The `--reload` flag enables auto-reloading during development. Do not use it in production. The port defaults to `8002` here based on the example env var `LOCAL_PORT`.
    ```bash
    # Ensure LOCAL_PORT is set in your environment or .env file
    uvicorn main_langgraph_enhanced:app --host 0.0.0.0 --port ${LOCAL_PORT:-8002} --reload
    ```
    The server will be available at `http://localhost:8002` (or the specified `LOCAL_PORT`).

## Docker Usage

This is the recommended way to run the application in production or testing environments. The container runs the application on port `8000` internally (hardcoded in the Dockerfile CMD).

1.  **Build the Docker Image:**
    Ensure you are in the `FastAgentAPI` directory containing the `Dockerfile`.
    ```bash
    docker build -t fastagentapi .
    ```
    *(You can also tag it more specifically, e.g., `teachings/fastagentapi:latest`)*.

2.  **Environment Variables for Runtime:**
    The Docker image uses environment variables for configuration. You can override the defaults set in the Dockerfile using the `-e` flag with `docker run`. Key variables include:

    | Variable              | Default in Dockerfile        | Description                                                                    |
    | :-------------------- | :--------------------------- | :----------------------------------------------------------------------------- |
    | `BACKEND_API_URL`     | `http://localhost:8001/v1`   | **Required override:** The full URL (including `/v1`) of your backend LLM.     |
    | `BACKEND_API_KEY`     | `sk-1111`                    | Optional API key for the backend LLM. Set to `""` if not needed.               |
    | `LOG_FILE_PATH`       | `/app/logs/app.log`          | Path *inside the container* where the log file will be written.                |
    | `VERBOSE_LOGGING`     | `true`                       | Set to `false` to reduce console log verbosity (detailed logs still go to file). |
    | `DEFAULT_MAX_RETRIES` | `1`                          | Default number of retries for backend calls if request doesn't specify.        |
    | **`PORT`** | *(N/A)* | *The internal container port is fixed at **8000** in the Dockerfile `CMD`.* |

3.  **Running Containers:**
    You can run multiple instances of the application from the same image, each configured differently. Remember the container always listens on port `8000`.

    * **Create Host Directories for Logs (Recommended):**
        This allows you to easily access log files from your host machine.
        ```bash
        mkdir -p host_logs/instance1
        mkdir -p host_logs/instance2
        ```

    * **Example: Run Instance 1 (Host Port 8002):**
        This maps host port `8002` to the *fixed* container port `8000`, sets a specific backend URL, and configures a unique log file path mapped to the host.
        ```bash
        docker run -d --rm --name fastagentapi-instance1 \
          -p 8002:8000 \
          -e BACKEND_API_URL="http://192.168.1.11:8001/v1" \
          -e LOG_FILE_PATH="/app/logs/instance1.log" \
          -e BACKEND_API_KEY="<backend1_key_if_needed>" \
          -e VERBOSE_LOGGING="true" \
          -v $(pwd)/host_logs/instance1:/app/logs \
          fastagentapi
        ```
        *(On Windows Command Prompt, use `"%cd%/host_logs/instance1"` instead of `$(pwd)/host_logs/instance1` for the volume path)*.

    * **Example: Run Instance 2 (Host Port 8003):**
        This maps host port `8003` to the *fixed* container port `8000`, uses a *different* backend URL, and logs to a separate file, also mapped to the host.
        ```bash
        docker run -d --rm --name fastagentapi-instance2 \
          -p 10003:8000 \
          -e BACKEND_API_URL="http://192.168.1.11:10002/v1" \
          -e LOG_FILE_PATH="/app/logs/instance2.log" \
          -e BACKEND_API_KEY="<backend2_key_if_needed>" \
          -e VERBOSE_LOGGING="true" \
          -v $(pwd)/host_logs/instance2:/app/logs \
          fastagentapi
        ```
        *(On Windows Command Prompt, use `"%cd%/host_logs/instance2"` instead of `$(pwd)/host_logs/instance2` for the volume path)*.

    **Explanation of `docker run` flags:**
    * `-d`: Run container in detached mode (background).
    * `--name`: Assign a human-readable name to the container.
    * `-p <host_port>:<container_port>`: Map a port on the host to a port inside the container (container port is fixed at `8000`).
    * `-e <VAR_NAME>="<value>"`: Set an environment variable inside the container.
    * `-v <host_path>:<container_path>`: Mount a directory from the host into the container. This makes files written to `<container_path>` (like logs) appear in `<host_path>`.

4.  **Accessing Logs:**

    * **Console Logs:** View the logs streamed by the application (INFO/DEBUG level based on `VERBOSE_LOGGING`):
        ```bash
        docker logs fastagentapi-instance1
        docker logs fastagentapi-instance2
        # Add -f to follow logs in real-time
        docker logs -f fastagentapi-instance1
        ```

    * **Log Files (via Volume Mount):** Access the detailed log files directly on your host machine in the directories you created:
        * Instance 1: `host_logs/instance1/instance1.log`
        * Instance 2: `host_logs/instance2/instance2.log`

## API Endpoints

* **`POST /v1/chat/completions`**: The main OpenAI-compatible endpoint. Accepts standard OpenAI chat completion request bodies, including `messages`, `model`, `tools`, `response_format`, etc. Handled by the LangGraph workflow.
* **`GET /health`**: A simple health check endpoint. Returns `{"status": "ok"}`.
* **`/{path:path}`**: (Catch-all) Forwards any other requests directly to the `BACKEND_API_URL` base.

## Configuration Summary

The application is configured primarily through environment variables:

| Variable              | Purpose                                                            | Default (Dockerfile)         | Runtime Override Example                |
| :-------------------- | :----------------------------------------------------------------- | :--------------------------- | :-------------------------------------- |
| `BACKEND_API_URL`     | Backend LLM endpoint (full URL)                                    | `http://localhost:8001/v1`   | `-e BACKEND_API_URL="<url>"`            |
| `BACKEND_API_KEY`     | API Key for backend                                                | `sk-1111`                    | `-e BACKEND_API_KEY="<key>"`            |
| `LOG_FILE_PATH`       | Internal path for log file                                         | `/app/logs/app.log`          | `-e LOG_FILE_PATH="/app/logs/custom.log"` |
| `VERBOSE_LOGGING`     | Enable DEBUG level logging to console (`true` or `false`)          | `true`                       | `-e VERBOSE_LOGGING="false"`            |
| `DEFAULT_MAX_RETRIES` | Default backend call retry attempts                                | `1`                          | `-e DEFAULT_MAX_RETRIES=3`              |
| *`PORT`* | *Internal container port is fixed at 8000 in the Docker `CMD`* | *(Fixed: 8000)* | *(Not configurable via `-e PORT=...`)* |

---

## Optional: Graph Visualization (`flow.png`)

The application includes code to generate a visual representation of the LangGraph workflow, saving it as `flow.png` in the application's root directory upon startup. Generating this graph requires the `pygraphviz` Python library and the underlying `graphviz` system dependency.

This is **not** installed in the Docker container by default. If you wish to generate the graph, you should install these dependencies locally and run the Python script directly:

**1. Install Graphviz System Dependency:**

* **macOS (using Homebrew):**
    ```bash
    brew install graphviz
    ```
* **Debian/Ubuntu Linux (using apt):**
    ```bash
    sudo apt-get update
    sudo apt-get install graphviz libgraphviz-dev -y
    ```
* **Fedora/CentOS/RHEL Linux (using dnf/yum):**
    ```bash
    sudo dnf install graphviz graphviz-devel -y
    # or using yum:
    # sudo yum install graphviz graphviz-devel -y
    ```
* **Windows:**
    * Download an installer from the official Graphviz download page: [https://graphviz.org/download/](https://graphviz.org/download/)
    * Ensure the Graphviz `bin` directory is added to your system's PATH environment variable during or after installation.

**2. Install `pygraphviz`:**

* Activate your Python virtual environment (if you created one for local development):
    ```bash
    # e.g., conda activate fastagentapi or source venv/bin/activate
    ```
* Install using pip:
    ```bash
    pip install pygraphviz
    ```
    *(Note: This might take some time as it often compiles parts of the library).*

**3. Run the Application Locally:**

* Ensure your necessary environment variables (`BACKEND_API_URL`, `LOCAL_PORT` etc.) are set locally (e.g., using a `.env` file and `python-dotenv`).
* Run the script directly:
    ```bash
    # This will trigger graph generation if pygraphviz is installed
    python main_langgraph_enhanced.py
    ```
    * Alternatively, run with uvicorn (graph generates on import/startup):
        ```bash
        uvicorn main_langgraph_enhanced:app --port ${LOCAL_PORT:-8002} --reload
        ```
* The `flow.png` file should now be generated in your project's root directory.