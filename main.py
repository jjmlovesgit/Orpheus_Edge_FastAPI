# main_fastapi.py

# --- Standard Library Imports ---
import logging
import os
import json
import re
import time
import traceback # Keep if used by any remaining part
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import warnings
import io
import sys
import shutil # STT: Added for file operations
import tempfile # STT: Added for temporary file management

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
# --- END LOGGING SETUP ---

# --- Third-Party Imports ---
import numpy as np
import requests
import torch
from torch import nn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File # STT: Added UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# --- Import the LLM router ---
from llm_router import router as llm_api_router
# --- End Import LLM router ---

# --- SNAC and Whisper Imports ---
try:
    from snac import SNAC
    logger.info("SNAC imported successfully for TTS.")
except ImportError:
    logger.error("SNAC not found. Please install it: pip install git+https://github.com/hubertsiuzdak/snac.git")
    SNAC = None

# STT: Attempt to import Whisper
try:
    import whisper
    logger.info("Whisper imported successfully for STT.")
except ImportError:
    logger.error("Whisper library not found. Please install it: pip install -U openai-whisper")
    whisper = None # Set to None if import fails
# --- End Imports ---

# --- Constants ---
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")

# TTS Constants
TTS_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/completions"
TTS_MODEL = os.getenv("TTS_MODEL", "orpheus-3b-0.1")
TTS_PROMPT_FORMAT = "<|audio|>{voice}: {text}<|eot_id|>"
TTS_PROMPT_STOP_TOKENS = ["<|eot_id|>", "<|audio|>"]
DEFAULT_TTS_TEMP = 2.0
DEFAULT_TTS_TOP_P = 0.9
DEFAULT_TTS_REP_PENALTY = 1.1 # fixed value
ORPHEUS_MIN_ID = 10
ORPHEUS_TOKENS_PER_LAYER = 4096
ORPHEUS_N_LAYERS = 7
ORPHEUS_MAX_ID = ORPHEUS_MIN_ID + (ORPHEUS_N_LAYERS * ORPHEUS_TOKENS_PER_LAYER)
TARGET_SAMPLE_RATE = 24000
TTS_STREAM_MIN_GROUPS = 5
TTS_STREAM_SILENCE_MS = 0
DEFAULT_MIN_DECODE_BATCH_GROUPS = 7
ALL_VOICES = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
DEFAULT_TTS_VOICE = ALL_VOICES[0]
TTS_FAILED_MSG = "(TTS generation failed or produced no audio)"

# STT Constants --- NEW ---
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en") # e.g., "tiny.en", "base.en", "small.en"
TEMP_AUDIO_DIR = "temp_stt_audio_files" # Temporary directory for STT audio uploads
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True) # Ensure temp directory exists
# --- End STT Constants ---

# Shared constants
STREAM_TIMEOUT_SECONDS = 300
STREAM_HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
SSE_DATA_PREFIX = "data:"
SSE_DONE_MARKER = "[DONE]"
# --- End Constants ---

# --- Device Setup ---
# This device will be used for both SNAC (TTS) and Whisper (STT)
# If you have multiple GPUs, you might want more sophisticated device selection.
tts_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Selected device for AI models: '{tts_device}'")
# --- End Device Setup ---

# --- Helper Functions (parse_gguf_codes, redistribute_codes, apply_fade - Unchanged) ---
def parse_gguf_codes(response_text: str) -> List[int]:
    """Parses GGUF-style custom audio tokens."""
    try:
        codes = [
            int(m) for m in re.findall(r"<custom_token_(\d+)>", response_text)
            if ORPHEUS_MIN_ID <= int(m) < ORPHEUS_MAX_ID
        ]
        return codes
    except Exception as e:
        logger.error(f"GGUF parse error: {e} on text: '{response_text[:200]}...'", exc_info=True)
        return []

def redistribute_codes(codes: List[int], model: nn.Module) -> Optional[np.ndarray]:
    """Redistributes parsed codes into SNAC layers and decodes using the SNAC model."""
    if not codes or model is None:
        logger.debug("Redistribute codes called with no codes or no model.")
        return None

    if len(codes) % ORPHEUS_N_LAYERS != 0:
        logger.warning(f"Redistribute codes: Received {len(codes)} codes which is not a multiple of ORPHEUS_N_LAYERS ({ORPHEUS_N_LAYERS}). Processing full groups only.")

    num_groups_in_input = len(codes) // ORPHEUS_N_LAYERS
    if num_groups_in_input == 0:
        logger.debug("Redistribute codes: Not enough codes for a complete group.")
        return None

    try:
        dev = next(model.parameters()).device
        snac_layers_codes: List[List[int]] = [[] for _ in range(3)] # SNAC expects 3 layers

        num_groups_to_process = num_groups_in_input
        codes_to_process = codes[:num_groups_to_process * ORPHEUS_N_LAYERS]
        logger.debug(f"Redistribute codes: Processing {len(codes_to_process)} codes as {num_groups_to_process} full groups.")

        valid_groups_count = 0
        for i in range(num_groups_to_process):
            idx = i * ORPHEUS_N_LAYERS
            group = codes_to_process[idx : idx + ORPHEUS_N_LAYERS]

            processed: List[Optional[int]] = [None] * ORPHEUS_N_LAYERS
            group_is_valid = True
            for j, t_id in enumerate(group):
                layer_idx_from_id = (t_id - ORPHEUS_MIN_ID) // ORPHEUS_TOKENS_PER_LAYER
                code_idx = (t_id - ORPHEUS_MIN_ID) % ORPHEUS_TOKENS_PER_LAYER
                expected_layer_for_pos = j

                if not (ORPHEUS_MIN_ID <= t_id < ORPHEUS_MAX_ID):
                    logger.warning(f"Redistribute codes: Invalid token ID {t_id} in group {i} at position {j}. Skipping group.")
                    group_is_valid = False; break
                if layer_idx_from_id != expected_layer_for_pos:
                    logger.warning(f"Redistribute codes: Code {t_id} (layer {layer_idx_from_id}) found at unexpected position {j} (expected layer {expected_layer_for_pos}) in group {i}. Skipping group.")
                    group_is_valid = False; break
                processed[j] = code_idx
            
            if group_is_valid:
                try:
                    if any(p is None for p in processed):
                        logger.error(f"Redistribute codes: 'None' found in processed list for a valid group {i}. Skipping.")
                        continue
                    
                    pg_int: List[int] = [p for p in processed if p is not None] 
                    if len(pg_int) != ORPHEUS_N_LAYERS: 
                         logger.error(f"Redistribute codes: Group {i} valid, but processed list length mismatch. Skipping")
                         continue

                    snac_layers_codes[0].append(pg_int[0]) 
                    snac_layers_codes[1].append(pg_int[1]) 
                    snac_layers_codes[2].append(pg_int[2]) 
                    snac_layers_codes[2].append(pg_int[3]) 
                    snac_layers_codes[1].append(pg_int[4]) 
                    snac_layers_codes[2].append(pg_int[5]) 
                    snac_layers_codes[2].append(pg_int[6]) 
                    valid_groups_count += 1
                except IndexError as map_e:
                    logger.error(f"Redistribute codes: Code mapping error in group {i}, processed={processed}: {map_e}. Skipping group.", exc_info=True)
                    continue
                except TypeError as type_e: 
                    logger.error(f"Redistribute codes: Type error during mapping (likely None in processed) in group {i}, processed={processed}: {type_e}. Skipping group.", exc_info=True)
                    continue

        if valid_groups_count == 0:
            logger.warning("Redistribute codes: No valid Orpheus code groups could be mapped to SNAC layers.")
            return None

        expected_l0_count = valid_groups_count
        expected_l1_count = valid_groups_count * 2
        expected_l2_count = valid_groups_count * 4

        if len(snac_layers_codes[0]) != expected_l0_count or \
           len(snac_layers_codes[1]) != expected_l1_count or \
           len(snac_layers_codes[2]) != expected_l2_count:
            logger.error(f"Redistribute codes: Mismatched SNAC input tensor lengths after mapping {valid_groups_count} valid groups: L0={len(snac_layers_codes[0])}, L1={len(snac_layers_codes[1])}, L2={len(snac_layers_codes[2])}. Expected: {expected_l0_count}, {expected_l1_count}, {expected_l2_count}. Aborting decode.")
            return None
        
        tensors = [ torch.tensor(lc, device=dev, dtype=torch.long).unsqueeze(0) for lc in snac_layers_codes ]
        logger.debug(f"Redistribute codes: Decodable group count: {valid_groups_count}. Decoding...")
        with torch.no_grad():
            audio = model.decode(tensors) # type: ignore
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        audio_np = audio.detach().squeeze().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(audio_np)):
            logger.error("!!! SNAC produced non-finite values (NaN/inf). Replacing with silence. !!!")
            return np.zeros_like(audio_np)
        logger.debug(f"Redistribute codes: Decode successful. Generated {audio_np.size} samples.")
        return audio_np
    except Exception as e:
        logger.exception(f"SNAC decode error: {e}")
        return None

def apply_fade(audio_chunk: np.ndarray, sample_rate: int, fade_ms: int = 5) -> np.ndarray:
    """Applies a fade in/out to an audio chunk."""
    if audio_chunk is None or audio_chunk.size == 0:
        return audio_chunk
    num_fade_samples = int(sample_rate * (fade_ms / 1000.0))
    if num_fade_samples <= 0 or audio_chunk.size < 2 * num_fade_samples:
        return audio_chunk 
    fade_in = np.linspace(0., 1., num_fade_samples, dtype=audio_chunk.dtype)
    fade_out = np.linspace(1., 0., num_fade_samples, dtype=audio_chunk.dtype)
    audio_chunk[:num_fade_samples] *= fade_in
    audio_chunk[-num_fade_samples:] *= fade_out
    logger.debug(f"Applied {fade_ms}ms fade to audio chunk.")
    return audio_chunk
# --- End Helper Functions ---

# --- Model Loading ---
logger.info("--- Loading AI Models ---")

# SNAC Model for TTS
snac_model: Optional[SNAC] = None
if SNAC is not None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="snac.snac")
            snac_model_instance = None
            try:
                snac_model_instance = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
                logger.info("SNAC.from_pretrained called successfully.")
            except Exception as load_e:
                logger.error(f"SNAC.from_pretrained failed: {load_e}", exc_info=True)

        if snac_model_instance:
            snac_model = snac_model_instance.to(tts_device).eval() # type: ignore
            logger.info(f"SNAC model loaded successfully to '{tts_device}'.")
        else:
            logger.error("SNAC.from_pretrained returned None or failed. SNAC model not loaded.")
            snac_model = None
    except Exception as e:
        logger.exception("Fatal error during SNAC model loading process.")
        snac_model = None
if not snac_model:
    logger.critical("SNAC model failed to load. TTS will be unavailable.")

# Whisper Model for STT 
whisper_model: Optional[Any] = None 
if whisper is not None: 
    try:
        logger.info(f"Loading Whisper model ('{WHISPER_MODEL_NAME}') onto '{tts_device}'...")
        with warnings.catch_warnings(): 
            warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
            if tts_device == "cpu":
                warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
        
        whisper_model_instance = whisper.load_model(WHISPER_MODEL_NAME, device=tts_device)
        whisper_model = whisper_model_instance
        logger.info(f"Whisper STT model ('{WHISPER_MODEL_NAME}') loaded successfully to '{tts_device}'.")
    except Exception as e:
        logger.exception(f"Fatal error during Whisper STT model ('{WHISPER_MODEL_NAME}') loading.")
        whisper_model = None
if not whisper_model:
    logger.warning("Whisper STT model failed to load. STT functionality will be unavailable.")
# --- End Model Loading ---

# --- TTS Stream Generator ---
def generate_speech_stream_bytes(
    text: str, voice: str, tts_temperature: float, tts_top_p: float,
    tts_repetition_penalty: float, buffer_groups_param: int, padding_ms_param: int,
    min_decode_batch_groups_param: int
) -> Generator[bytes, None, None]:
    global snac_model 
    if not text.strip():
        logger.warning("generate_speech_stream_bytes called with empty text.")
        return
    if snac_model is None:
        logger.error("SNAC model not loaded. Cannot generate audio bytes for TTS. Check server logs.")
        yield b'' 
        return

    codes_per_group = ORPHEUS_N_LAYERS
    buffer_groups_effective = max(1, buffer_groups_param)
    min_codes_required_initial = buffer_groups_effective * codes_per_group
    min_decode_batch_groups = max(1, min_decode_batch_groups_param)
    min_codes_required_per_batch = min_decode_batch_groups * codes_per_group
    min_codes_required_per_batch = max(min_codes_required_per_batch, codes_per_group) 

    logger.debug(f"TTS Stream processing: initial buffer target={buffer_groups_effective} groups ({min_codes_required_initial} codes), padding={padding_ms_param} ms. Per-batch decode unit: {min_codes_required_per_batch} codes ({min_decode_batch_groups} groups).")

    silence_bytes = b''
    if padding_ms_param > 0:
        silence_samples = int(TARGET_SAMPLE_RATE * (padding_ms_param / 1000.0))
        if silence_samples > 0:
            logger.debug(f"Calculated silence samples per side: {silence_samples}")
            silence_bytes = np.zeros(silence_samples, dtype=np.float32).tobytes()

    payload = {
        "model": TTS_MODEL, "prompt": TTS_PROMPT_FORMAT.format(voice=voice, text=text),
        "temperature": tts_temperature, "top_p": tts_top_p, "repeat_penalty": tts_repetition_penalty,
        "n_predict": -1, "stop": TTS_PROMPT_STOP_TOKENS, "stream": True
    }
    accumulated_codes: List[int] = []
    request_initiation_time = time.time() # Renamed from stream_start_time for clarity with TTFT
    response_obj = None
    stream_successful = False
    initial_buffer_processed = False

    # --- ADDED: Variables for TTFT tracking ---
    time_stream_connected_for_ttft: Optional[float] = None
    first_token_parsed_time_for_ttft: Optional[float] = None
    calculated_ttft_ms: Optional[float] = None
    # --- END ADDED ---

    try:
        logger.info(">>> TTS API: Initiating stream request to external TTS server...")
        logger.debug(f"Sending TTS Payload: {json.dumps(payload)}")
        response_obj = requests.post(
            TTS_API_ENDPOINT, json=payload, headers=STREAM_HEADERS, stream=True, timeout=STREAM_TIMEOUT_SECONDS
        )
        response_obj.raise_for_status()
        
        # --- MODIFIED: Capture time for TTFT baseline and use it in the existing log ---
        time_stream_connected_for_ttft = time.time() # This is the point after connection is made, before reading lines
        # Original log line used time.time() directly, now uses the captured variable for TTFT baseline
        logger.info(f"--- TTS API: Stream connected successfully after {time_stream_connected_for_ttft - request_initiation_time:.3f}s. Receiving codes...")
        # --- END MODIFIED ---

        for line in response_obj.iter_lines():
            # Using time_stream_connected_for_ttft as the start for timeout once connected
            # This is safer if request_initiation_time is much earlier than actual connection
            timeout_baseline = time_stream_connected_for_ttft if time_stream_connected_for_ttft is not None else request_initiation_time
            if time.time() - timeout_baseline > STREAM_TIMEOUT_SECONDS: # Check against when we *started receiving*
                logger.error(f"❌ TTS API stream processing timed out after {STREAM_TIMEOUT_SECONDS} seconds while waiting for data.")
                break
            if not line: continue

            try:
                decoded_line = line.decode(response_obj.encoding or 'utf-8', errors='ignore')
            except UnicodeDecodeError:
                logger.warning(f"Skipping undecodable line in TTS stream: {line[:50]}..."); continue
            
            if decoded_line.startswith(SSE_DATA_PREFIX):
                json_str = decoded_line[len(SSE_DATA_PREFIX):].strip()
                if json_str == SSE_DONE_MARKER:
                    logger.debug("Received TTS SSE_DONE_MARKER. Ending stream processing."); break
                if not json_str:
                    logger.debug("Received empty data line, skipping."); continue

                try:
                    data = json.loads(json_str)
                    chunk_text = ""
                    if "content" in data:
                        chunk_text = data.get("content", "")
                    elif "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        chunk_text = delta.get("content", "") or choice.get("text", "")

                    if chunk_text:
                        new_codes = parse_gguf_codes(chunk_text)
                        if new_codes:
                            # --- ADDED: Logic to capture time of first token chunk ---
                            if first_token_parsed_time_for_ttft is None and time_stream_connected_for_ttft is not None:
                                first_token_parsed_time_for_ttft = time.time()
                                calculated_ttft_ms = (first_token_parsed_time_for_ttft - time_stream_connected_for_ttft) * 1000
                                logger.info(f"--- TTS METRIC: Time to first audio token chunk parsed: {calculated_ttft_ms:.2f} ms (from stream connected) ---")
                            # --- END ADDED ---
                            
                            accumulated_codes.extend(new_codes)
                            #logger.debug(f"Parsed {len(new_codes)} new codes. Total accumulated: {len(accumulated_codes)}.") # Your original log

                            if not initial_buffer_processed and len(accumulated_codes) >= min_codes_required_initial:
                                logger.debug(f"Have {len(accumulated_codes)} codes, >= initial buffer {min_codes_required_initial}. Attempting to decode initial buffer batch.")
                                num_groups_for_initial_buffer = min_codes_required_initial // codes_per_group 
                                codes_to_decode_count = num_groups_for_initial_buffer * codes_per_group
                                
                                if len(accumulated_codes) >= codes_to_decode_count and codes_to_decode_count > 0 :
                                    codes_to_decode_initial = accumulated_codes[:codes_to_decode_count]
                                    accumulated_codes = accumulated_codes[codes_to_decode_count:]
                                    logger.debug(f"Decoding initial {len(codes_to_decode_initial)} codes. Remaining accumulated: {len(accumulated_codes)}")
                                    try:
                                        snac_decode_start_time = time.time()
                                        audio_chunk = redistribute_codes(codes_to_decode_initial, snac_model)
                                        snac_decode_end_time = time.time()
                                        if audio_chunk is not None and audio_chunk.size > 0:
                                            logger.debug(f"--- SNAC: Decoded initial buffer ({len(codes_to_decode_initial)} codes -> {audio_chunk.size} samples) in {snac_decode_end_time - snac_decode_start_time:.3f}s.")
                                            faded_chunk = apply_fade(audio_chunk, TARGET_SAMPLE_RATE, fade_ms=3)
                                            audio_bytes_to_yield = faded_chunk.astype(np.float32).tobytes()
                                            if silence_bytes: yield silence_bytes 
                                            yield audio_bytes_to_yield
                                            stream_successful = True
                                            logger.info("Initial buffer decoded and yielded. Switching to per-batch decoding.")
                                        else:
                                            logger.warning(f"--- SNAC: Failed to decode initial buffer ({len(codes_to_decode_initial)} codes) in {snac_decode_end_time - snac_decode_start_time:.3f}s, or produced no audio.")
                                    except Exception as decode_e:
                                        logger.exception(f"Error during initial buffer decoding/yielding: {decode_e}.")
                                    finally:
                                        initial_buffer_processed = True 
                                else:
                                     logger.debug(f"Not enough codes ({len(accumulated_codes)}) for the calculated initial batch size ({codes_to_decode_count}), or batch size is zero. Waiting for more codes or initial_buffer_processed to be true.")

                            while initial_buffer_processed and len(accumulated_codes) >= min_codes_required_per_batch:
                                logger.debug(f"Entering per-batch while loop. Have {len(accumulated_codes)} codes, >= {min_codes_required_per_batch}. Processing batches.")
                                try:
                                    num_batches_to_decode = len(accumulated_codes) // min_codes_required_per_batch
                                    codes_to_decode_count_batch = num_batches_to_decode * min_codes_required_per_batch
                                    
                                    if codes_to_decode_count_batch == 0: 
                                        logger.warning("Per-batch: codes_to_decode_count_batch is zero, breaking while loop.")
                                        break

                                    codes_to_decode_batch = accumulated_codes[:codes_to_decode_count_batch]
                                    accumulated_codes = accumulated_codes[codes_to_decode_count_batch:]
                                    logger.debug(f"Decoding {len(codes_to_decode_batch)} codes in {num_batches_to_decode} batches. Remaining accumulated codes: {len(accumulated_codes)}.")
                                    
                                    snac_decode_start_time = time.time()
                                    audio_chunk = redistribute_codes(codes_to_decode_batch, snac_model)
                                    snac_decode_end_time = time.time()

                                    if audio_chunk is not None and audio_chunk.size > 0:
                                        logger.debug(f"--- SNAC: Decoded chunk ({len(codes_to_decode_batch)} codes -> {audio_chunk.size} samples) in {snac_decode_end_time - snac_decode_start_time:.3f}s.")
                                        faded_chunk = apply_fade(audio_chunk, TARGET_SAMPLE_RATE, fade_ms=3)
                                        audio_bytes_to_yield = faded_chunk.astype(np.float32).tobytes()
                                        if silence_bytes: yield silence_bytes 
                                        yield audio_bytes_to_yield
                                        stream_successful = True
                                    else:
                                        logger.warning(f"--- SNAC: Failed to decode chunk ({len(codes_to_decode_batch)} codes) in {snac_decode_end_time - snac_decode_start_time:.3f}s, or produced no audio. Skipping yield for this batch.")
                                except Exception as decode_e:
                                    logger.exception(f"Error during per-batch decoding/yielding: {decode_e}. Skipping this batch.")
                    
                    stop_reason = None
                    if "choices" in data and data["choices"] and data["choices"][0].get("finish_reason"):
                        stop_reason = data["choices"][0].get("finish_reason")
                        logger.debug(f"TTS Stream stop condition met from API response: reason='{stop_reason}'")
                        break 
                    if data.get("stop") is True or data.get("stopped_eos") is True:
                        logger.debug(f"TTS Stream stop condition met from API data flags: stop={data.get('stop')}, stopped_eos={data.get('stopped_eos')}")
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in TTS stream line: '{json_str[:100]}...'", exc_info=False); continue
                except Exception as e_proc:
                    logger.exception(f"Error processing TTS stream chunk data: '{json_str[:100]}...'")
                    break 

        logger.debug(f"TTS Stream ended or timed out. Processing final accumulated codes ({len(accumulated_codes)}).")
        if len(accumulated_codes) >= codes_per_group: 
            num_final_groups = len(accumulated_codes) // codes_per_group
            codes_to_decode_final = accumulated_codes[:num_final_groups * codes_per_group]
            logger.debug(f"Decoding final {len(codes_to_decode_final)} codes in {num_final_groups} groups.")
            snac_decode_start_time = time.time()
            audio_chunk = redistribute_codes(codes_to_decode_final, snac_model)
            snac_decode_end_time = time.time()
            if audio_chunk is not None and audio_chunk.size > 0:
                logger.debug(f"--- SNAC: Decoded final chunk ({len(codes_to_decode_final)} codes -> {audio_chunk.size} samples) in {snac_decode_end_time - snac_decode_start_time:.3f}s.")
                faded_chunk = apply_fade(audio_chunk, TARGET_SAMPLE_RATE, fade_ms=3)
                audio_bytes_to_yield = faded_chunk.astype(np.float32).tobytes()
                if silence_bytes: yield silence_bytes 
                yield audio_bytes_to_yield
                if silence_bytes: yield silence_bytes 
                stream_successful = True
            else:
                logger.warning(f"--- SNAC: Failed to decode final chunk ({len(codes_to_decode_final)} codes) in {snac_decode_end_time - snac_decode_start_time:.3f}s, or produced no audio.")
        elif accumulated_codes:
            logger.debug(f"Discarding final {len(accumulated_codes)} codes (less than {codes_per_group}) after stream end.")

    except requests.exceptions.Timeout:
        logger.error(f"❌ TTS API stream request timed out after {STREAM_TIMEOUT_SECONDS} seconds.", exc_info=True)
    except requests.exceptions.RequestException as req_e:
        logger.exception(f"❌ TTS API stream request failed: {req_e}")
    except Exception as e:
        logger.exception(f"❌ Unexpected error during TTS stream processing loop: {e}")
    finally:
        if response_obj:
            logger.debug("Closing TTS API response connection.")
            try:
                response_obj.close()
            except Exception as close_e:
                logger.warning(f"Error closing requests response: {close_e}")

    # --- MODIFIED: Final Summary Log to include TTFT ---
    if not stream_successful:
        logger.error("--- TTS Stream Generation Finished - FAILED TO PRODUCE ANY AUDIO ---")
    else:
        ttft_summary_msg = f"(TTFT: {calculated_ttft_ms:.2f} ms)" if calculated_ttft_ms is not None else "(TTFT not recorded, e.g. no codes received)"
        logger.info(f"--- First Chunk TTS Stream Generation Finished Successfully {ttft_summary_msg} ---")
    # --- END MODIFIED ---
# --- End TTS Stream Generator ---

# --- FastAPI App Setup ---
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static_assets")

# --- Include the LLM router ---
app.include_router(llm_api_router, prefix="/api/llm", tags=["LLM"])
# --- End Include LLM router ---

# --- Pydantic Models for TTS and STT ---
class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_TTS_VOICE
    tts_temperature: float = DEFAULT_TTS_TEMP
    tts_top_p: float = DEFAULT_TTS_TOP_P
    tts_repetition_penalty: float = DEFAULT_TTS_REP_PENALTY
    buffer_groups: int = TTS_STREAM_MIN_GROUPS
    padding_ms: int = TTS_STREAM_SILENCE_MS
    min_decode_batch_groups: int = DEFAULT_MIN_DECODE_BATCH_GROUPS

# STT: Pydantic model for STT response --- NEW ---
class STTResponse(BaseModel):
    text: str
    language: Optional[str] = None
    error: Optional[str] = None
# --- End Pydantic Models ---

# --- FastAPI Endpoints ---
@app.post("/api/tts/stream")
async def tts_stream_endpoint(request: TTSRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] TTS: 🚀 Received POST request to /api/tts/stream")
    logger.info(f"[{request_id}] TTS: 📚 TTS Request Payload: {request.model_dump_json(indent=2)}")

    if snac_model is None:
        logger.error(f"[{request_id}] TTS: 🚫 SNAC model not loaded. Returning 503 error.")
        raise HTTPException(status_code=503, detail="SNAC model not loaded. TTS is unavailable.")

    audio_generator = generate_speech_stream_bytes(
        text=request.text, voice=request.voice, tts_temperature=request.tts_temperature,
        tts_top_p=request.tts_top_p, tts_repetition_penalty=request.tts_repetition_penalty,
        buffer_groups_param=request.buffer_groups, padding_ms_param=request.padding_ms,
        min_decode_batch_groups_param=request.min_decode_batch_groups
    )
    headers = {"X-Sample-Rate": str(TARGET_SAMPLE_RATE), "X-Audio-Format": "FLOAT32_PCM"}
    return StreamingResponse(audio_generator, media_type="audio/octet-stream", headers=headers)

# STT: New endpoint for Speech-to-Text --- NEW ---
@app.post("/api/stt/transcribe", response_model=STTResponse)
async def stt_transcribe_endpoint(audio_file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] STT: Received POST request to /api/stt/transcribe for file: {audio_file.filename}")

    if whisper_model is None:
        logger.error(f"[{request_id}] STT: Whisper model not loaded.")
        return STTResponse(text="", error="STT service unavailable: Whisper model not loaded.")

    tmp_audio_file_path = None 
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=TEMP_AUDIO_DIR, suffix=os.path.splitext(audio_file.filename)[1] or ".wav") as tmp_file:
            shutil.copyfileobj(audio_file.file, tmp_file)
            tmp_audio_file_path = tmp_file.name
        
        logger.info(f"[{request_id}] STT: Audio file '{audio_file.filename}' saved temporarily to '{tmp_audio_file_path}'")
        
        stt_start_time = time.time()
        result = whisper_model.transcribe(tmp_audio_file_path, fp16=(tts_device=="cuda")) # type: ignore
        stt_duration = time.time() - stt_start_time
        
        transcribed_text = result["text"].strip()
        detected_language = result.get("language", "unknown") 

        logger.info(f"[{request_id}] STT: Transcription successful in {stt_duration:.3f}s. Language: '{detected_language}'. Text: '{transcribed_text[:100]}...'")
        return STTResponse(text=transcribed_text, language=detected_language)

    except Exception as e:
        logger.exception(f"[{request_id}] STT: Error during transcription for file '{audio_file.filename}'")
        return STTResponse(text="", error=f"Transcription failed: {str(e)}")
    finally:
        if tmp_audio_file_path and os.path.exists(tmp_audio_file_path):
            try:
                os.remove(tmp_audio_file_path)
                logger.info(f"[{request_id}] STT: Cleaned up temporary audio file: {tmp_audio_file_path}")
            except Exception as cleanup_e:
                logger.warning(f"[{request_id}] STT: Failed to clean up temporary audio file '{tmp_audio_file_path}': {cleanup_e}")
        await audio_file.close()

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/script.js")
async def serve_script():
    return FileResponse("static/script.js", media_type="application/javascript")
# --- End FastAPI Endpoints ---

# --- Uvicorn Server ---
if __name__ == "__main__":
    if snac_model is None:
        logger.critical("SNAC Model not loaded at startup. TTS will be unavailable.")
    if whisper_model is None: 
        logger.warning("Whisper STT Model not loaded at startup. STT will be unavailable.")
    
    logger.info("Starting FastAPI Server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("FastAPI Server Stopped.")