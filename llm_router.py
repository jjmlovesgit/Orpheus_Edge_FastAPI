# llm_router.py
import logging
import os
import json
import uuid
import time
from typing import Any, Dict, Generator, List, Optional

import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__) # Will inherit root logger's config from main_fastapi.py

# --- LLM Constants ---
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")
LMSTUDIO_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/chat/completions"
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "dolphin3.0-llama3.1-8b-abliterated") # Default from user's Gradio script
LMSTUDIO_SYSTEM_PROMPT = os.getenv(
    "LMSTUDIO_SYSTEM_PROMPT",
    "You are engaged in a consensual adult chat with you ships log.  *** Important *** Strictly limit your responses to 120 or fewer words of two or three sentences. You are Captain Ersa Minor an the Captain of an interstellar start ship.  Your goal is to tranpost a group of 28 human survivors from earth to K2-18b an exoplanet 27 light years from earth.  You are crerating a story.  All topics are in scope and you remain in Charactor while playing off the chat responses from the ships AI to weave stories together."
)
DEFAULT_LMSTUDIO_MAX_TOKENS = -1
DEFAULT_LMSTUDIO_TEMP = 0.7
DEFAULT_LMSTUDIO_TOP_P = 0.9
DEFAULT_LMSTUDIO_TOP_K = int(os.getenv("DEFAULT_LMSTUDIO_TOP_K", "45")) # Ensure it's int, from Gradio script
DEFAULT_LMSTUDIO_REP_PENALTY = 1.1
LLM_CONTEXT_TURN_LIMIT = 3

# Shared Constants needed by LLM logic (copied here for encapsulation within the router)
STREAM_TIMEOUT_SECONDS = 300
STREAM_HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
SSE_DATA_PREFIX = "data:"
SSE_DONE_MARKER = "[DONE]"
LLM_FAILED_PREFIX = "[Error"
# --- End LLM Constants ---

# --- Pydantic Model for LLM Request ---
class LLMChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []
    temperature: float = DEFAULT_LMSTUDIO_TEMP
    top_p: float = DEFAULT_LMSTUDIO_TOP_P
    max_tokens: int = DEFAULT_LMSTUDIO_MAX_TOKENS
    repetition_penalty: float = DEFAULT_LMSTUDIO_REP_PENALTY
    top_k: Optional[int] = DEFAULT_LMSTUDIO_TOP_K
# --- End Pydantic Model ---

# --- LLM Stream Generator Function ---
def generate_llm_text_stream(
    prompt: str,
    history: List[Dict[str, str]],
    llm_temperature: float,
    llm_top_p: float,
    llm_max_tokens: int,
    llm_repetition_penalty: float,
    llm_top_k: Optional[int] = None
) -> Generator[str, None, None]:
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] LLM Router: Initiating LLM stream request.")

    messages = [{"role": "system", "content": LMSTUDIO_SYSTEM_PROMPT}]
    if history:
        limited_history = history[-(LLM_CONTEXT_TURN_LIMIT * 2):] # Send N*2 previous messages for N turns
        messages.extend(limited_history)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": messages,
        "temperature": llm_temperature,
        "top_p": llm_top_p,
        "max_tokens": llm_max_tokens if llm_max_tokens != -1 else None, # API might expect null for no limit
        "repeat_penalty": llm_repetition_penalty,
        "stream": True
    }
    if llm_top_k is not None and llm_top_k > 0: # top_k=0 often means disabled
        payload["top_k"] = llm_top_k
    
    # Remove None values from payload, especially if max_tokens became None
    payload = {k: v for k, v in payload.items() if v is not None}

    logger.debug(f"[{request_id}] LLM Router: Sending LLM Payload: {json.dumps(payload, indent=2)}")
    response_obj = None
    stream_start_time = time.time()

    try:
        response_obj = requests.post(
            LMSTUDIO_API_ENDPOINT, json=payload, headers=STREAM_HEADERS, stream=True, timeout=STREAM_TIMEOUT_SECONDS
        )
        response_obj.raise_for_status() # Check for HTTP errors (4xx or 5xx)
        logger.info(f"[{request_id}] LLM Router: LLM API Stream connected after {time.time() - stream_start_time:.3f}s.")
        error_occurred_in_stream = False

        for line in response_obj.iter_lines():
            if error_occurred_in_stream: break # Stop if an error was already processed from the stream
            if not line: continue

            try:
                decoded_line = line.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(f"[{request_id}] LLM Router: Skipping undecodable line in LLM stream: {line[:50]}..."); continue

            if decoded_line.startswith(SSE_DATA_PREFIX):
                json_str = decoded_line[len(SSE_DATA_PREFIX):].strip()
                if json_str == SSE_DONE_MARKER:
                    logger.debug(f"[{request_id}] LLM Router: Received LLM SSE_DONE_MARKER."); break
                if not json_str: continue

                try:
                    data = json.loads(json_str)
                    delta_content = None
                    finish_reason = None

                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "delta" in choice and choice["delta"] is not None:
                             delta_content = choice["delta"].get("content")
                        finish_reason = choice.get("finish_reason")
                    
                    if delta_content is not None:
                        yield delta_content
                    elif "error" in data: # Check for error objects in the stream
                        error_msg_detail = data.get('error', {}).get('message', 'Unknown LLM error from stream data')
                        formatted_error = f"{LLM_FAILED_PREFIX} (Stream Error: {error_msg_detail})"
                        logger.error(f"[{request_id}] LLM Router: {formatted_error}")
                        yield formatted_error; error_occurred_in_stream = True; break
                    
                    if finish_reason:
                        logger.debug(f"[{request_id}] LLM Router: LLM stream finish_reason: {finish_reason}")
                        # Typically, SSE_DONE_MARKER is the primary signal to stop.
                        # Some models might send finish_reason and then SSE_DONE.
                        # If finish_reason indicates an error or abnormal stop, might need specific handling.
                        if finish_reason not in ["stop", "length", None]: # "length" is a common valid reason
                             logger.warning(f"[{request_id}] LLM Router: Stream finished with non-standard reason: {finish_reason}")
                             # Depending on API, may want to break here or yield an indicator.

                except json.JSONDecodeError:
                    logger.warning(f"[{request_id}] LLM Router: Skipping invalid JSON in LLM stream: {json_str[:100]}..."); continue
                except Exception as e_proc:
                    logger.exception(f"[{request_id}] LLM Router: Error processing LLM stream chunk data: '{json_str[:100]}...'")
                    yield f"{LLM_FAILED_PREFIX} (Processing Error: {str(e_proc)})"; error_occurred_in_stream = True; break
        logger.info(f"[{request_id}] LLM Router: LLM stream processing loop finished after {time.time() - stream_start_time:.3f}s.")

    except requests.exceptions.Timeout:
        logger.error(f"[{request_id}] LLM Router: ❌ LLM API stream request timed out after {STREAM_TIMEOUT_SECONDS} seconds.", exc_info=True)
        yield f"{LLM_FAILED_PREFIX} (LLM stream request timed out)"
    except requests.exceptions.RequestException as req_e:
        logger.exception(f"[{request_id}] LLM Router: ❌ LLM API stream request failed: {req_e}")
        err_yield = f"{LLM_FAILED_PREFIX} (Error connecting to LLM server)"
        if hasattr(req_e, 'response') and req_e.response is not None:
            try:
                err_json = req_e.response.json()
                detail = err_json.get('error',{}).get('message') or err_json.get('detail', req_e.response.text)
                err_yield = f"{LLM_FAILED_PREFIX} (LLM Server Error: {str(detail)[:200]})" # Truncate potentially long errors
            except json.JSONDecodeError: # If error response is not JSON
                err_yield = f"{LLM_FAILED_PREFIX} (LLM Server Error: {req_e.response.status_code} - {req_e.response.text[:200]})"
        yield err_yield
    except Exception as e: # Catch-all for any other unexpected errors
        logger.exception(f"[{request_id}] LLM Router: ❌ Unexpected error during LLM stream generation: {e}")
        yield f"{LLM_FAILED_PREFIX} (Unexpected Error in LLM stream: {str(e)})"
    finally:
        if response_obj:
            response_obj.close()
            logger.debug(f"[{request_id}] LLM Router: Closed LLM API response connection.")
# --- End LLM Stream Generator ---

# --- APIRouter instance ---
router = APIRouter()
# --- End APIRouter instance ---

# --- LLM Chat Endpoint Definition ---
@router.post("/chat/stream", summary="Stream LLM Chat Completions", tags=["LLM"])
async def llm_chat_stream_endpoint_router(request_data: LLMChatRequest):
    # request_id can be generated here or use a middleware to add to request state if needed more broadly
    # For now, generate_llm_text_stream handles its own internal request_id for logging.
    # logger.info(f"LLM Router: 🚀 Received POST request to /chat/stream") # Logged within generate_llm_text_stream
    # logger.info(f"LLM Router: 📚 Request Payload: {request_data.model_dump_json(indent=2)}") # Logged within generate_llm_text_stream
    
    text_generator = generate_llm_text_stream(
        prompt=request_data.prompt,
        history=request_data.history,
        llm_temperature=request_data.temperature,
        llm_top_p=request_data.top_p,
        llm_max_tokens=request_data.max_tokens,
        llm_repetition_penalty=request_data.repetition_penalty,
        llm_top_k=request_data.top_k
    )
    return StreamingResponse(text_generator, media_type="text/event-stream")
# --- End LLM Chat Endpoint Definition ---