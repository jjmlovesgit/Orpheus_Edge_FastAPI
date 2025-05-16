![image](https://github.com/user-attachments/assets/dc45ee6a-c7bc-4d69-8460-13aaa37c34d6)


# Orpheus-Edge: Voice & Text AI Application

A FastAPI-based web application that provides a user interface for interacting with a Large Language Model (LLM), generating speech from text (TTS) using an Orpheus-compatible model and SNAC vocoder, and transcribing speech to text (STT) using Whisper.

## Features

* Real-time streaming of LLM responses.
* Streaming Text-to-Speech (TTS) generation.
* Speech-to-Text (STT) transcription using local Whisper models.
* Configurable parameters for LLM and TTS via the UI.
* Push-to-talk functionality using the space bar.
* Dark mode interface.

* ![image](https://github.com/user-attachments/assets/6b3cb773-6552-42d1-ac57-a410b40095bc)


## Prerequisites

* **Python:** Version 3.8 or higher is recommended.
* **pip:** Python package installer.
* **Git:** For cloning the repository.
* **External AI Services:** This application acts as a frontend and orchestration layer. It requires separate, already running AI model services for LLM responses and initial TTS audio code generation.
    * An **LLM inference server** (e.g., LM Studio running an LLM of your choice).
    * An **Orpheus-compatible TTS model server** that provides GGUF-style audio codes via a completions API (e.g., LM Studio running an Orpheus TTS model).
    * By default, these are expected to be accessible at `http://127.0.0.1:1234`.
* **(Optional) NVIDIA GPU & CUDA:** For GPU acceleration of local PyTorch models (SNAC vocoder, Whisper STT). The application will fall back to CPU if a GPU is not available.
* **(Optional 7 Highly recommended) FFmpeg:** Whisper may require `ffmpeg` to be installed on the system and available in the PATH for robust audio file format support during STT.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/jjmlovesgit/Orpheus_Edge_FastAPI
    cd Orpheus_Edge_FastAPI
    ```

2.  **Create and Activate a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```
    * On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install Dependencies:**
    Make sure you have the `requirements.txt` file and then run:
    ```bash
    pip install -r requirements.txt
    ```
    This will install FastAPI, Uvicorn, PyTorch, NumPy, Requests, OpenAI-Whisper, SNAC, and other necessary packages.

4.  **Model Downloads (First Run):**
    * The SNAC vocoder model (for TTS) and the specified Whisper model (for STT) will be downloaded automatically by the application on its first run if they are not already cached locally by the respective libraries. An internet connection is required for this initial download.

## Configuration

The application uses environment variables for configuration, with sensible defaults provided in the code. You can set these variables in your shell before running the application, or via your deployment environment's configuration tools.

Key environment variables:

* **`SERVER_BASE_URL`**: The base URL for your external LLM and Orpheus-compatible TTS server.
    * Default: `http://127.0.0.1:1234`
* **`LMSTUDIO_MODEL`**: (Used by `llm_router.py`) The model identifier for your LLM on the external server.
    * Default: `"dolphin3.0-llama3.1-8b-abliterated"` (adjust as per your LLM server setup)
* **`LMSTUDIO_SYSTEM_PROMPT`**: (Used by `llm_router.py`) The system prompt for the LLM.
    * Default: (A specific story-crafting prompt - see `llm_router.py`)
* **`TTS_MODEL`**: The model identifier for your Orpheus-compatible TTS model on the external server.
    * Default: `"orpheus-3b-0.1"` (adjust as per your TTS server setup)
* **`WHISPER_MODEL_NAME`**: The Whisper model to use for STT (e.g., "tiny.en", "base.en", "small.en", "medium.en"). Larger models are more accurate but require more resources.
    * Default: `"base.en"`

## Running the External AI Services (Crucial Prerequisite!)

Before starting this FastAPI application, you **MUST** ensure that your external LLM server and your Orpheus-compatible TTS server are running and accessible at the URL specified by `SERVER_BASE_URL` (or its default `http://127.0.0.1:1234`).

* **LM Studio Example:** If using LM Studio, load your desired LLM and start the server (usually on port 1234).
* **Orpheus TTS Server Example:** If using a llama.cpp based server for an Orpheus TTS model, ensure it's running and serving the completions API endpoint.

This application will not function correctly if these backend AI services are not available.

## Running the Application

Once prerequisites are met, external AI services are running, and dependencies are installed:

1.  Navigate to the project's root directory (where `main.py` is located).
2.  Ensure your Python virtual environment is activated.
3.  To run the application, use the following command in your terminal:

    ```bash
    uvicorn main:app --reload
    ```

    * **`main`**: Refers to your Python file `main.py`.
    * **`app`**: Refers to the FastAPI instance you created in that file (e.g., `app = FastAPI()`).
    * **`--reload`**: This flag enables auto-reloading, which is very useful during development as the server will automatically restart when you make changes to your code. For production deployments, you would typically omit this flag.

## Accessing the Application

By default, when you run Uvicorn with the command `uvicorn main:app --reload` (without specifying a `--host` or `--port`), it binds to `127.0.0.1` (localhost) on port `8000`.

You can access the application by opening your web browser and navigating to:
**`http://127.0.0.1:8000`**

**Optional - Running on a different host or port:**

If you want to make the server accessible from other devices on your network or run it on a different port, you can specify the `--host` and `--port` arguments:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

Using --host 0.0.0.0 will make the server listen on all available network interfaces. You would then access it via your machine's IP address (e.g., http://<your-local-ip-address>:8000).

# Model Reference -- Important for stability and performance:

## Orpheus TTS Settings Reference screen shots (See right side of screens)
![image](https://github.com/user-attachments/assets/c0c54e7e-9b01-464b-9751-b5add023eb5f)
![image](https://github.com/user-attachments/assets/aa1fc9e7-2a39-4840-b90e-32b783882743)

## LLM Settings Reference screen shots (See right side of screens)
![image](https://github.com/user-attachments/assets/a4847185-6280-42a2-adec-25b070f4ccc7)
![image](https://github.com/user-attachments/assets/658cfbcb-ee42-4519-b391-a737bed48f23)



