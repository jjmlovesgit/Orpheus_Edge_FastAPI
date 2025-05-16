document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed");

    // --- UI Element References ---
    const appForm = document.getElementById('app-form');
    const textInput = document.getElementById('text_input');
    const generateButton = document.getElementById('generate-button');
    const audioPlayer = document.getElementById('audio-player');
    const modeSelect = document.getElementById('mode_select');
    const chatHistoryDisplay = document.getElementById('chat-history-display');
    const ttsVoiceSelect = document.getElementById('tts_voice_dd');
    const recordButton = document.getElementById('record-button');
    const recordStatus = document.getElementById('record-status');

    const ttsVoices = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"];
    const ttsSliders = [
        { id: 'tts_temp_slider', valueId: 'tts_temp_value', isFloat: true, precision: 2 },
        { id: 'tts_top_p_slider', valueId: 'tts_top_p_value', isFloat: true, precision: 2 },
        { id: 'tts_rep_penalty_slider', valueId: 'tts_rep_penalty_value', isFloat: true, precision: 2 },
        { id: 'tts_buffer_groups_slider', valueId: 'tts_buffer_groups_value', isFloat: false },
        { id: 'tts_padding_ms_slider', valueId: 'tts_padding_ms_value', isFloat: false },
        { id: 'tts_batch_groups_slider', valueId: 'tts_batch_groups_value', isFloat: false },
        { id: 'client_buffer_duration_slider', valueId: 'client_buffer_duration_value', isFloat: true, precision: 2 }
    ];
    const llmSliders = [
        { id: 'llm_temp_slider', valueId: 'llm_temp_value', isFloat: true, precision: 2 },
        { id: 'llm_top_p_slider', valueId: 'llm_top_p_value', isFloat: true, precision: 2 },
        { id: 'llm_rep_penalty_slider', valueId: 'llm_rep_penalty_value', isFloat: true, precision: 2 },
        { id: 'llm_top_k_slider', valueId: 'llm_top_k_value', isFloat: false }
    ];
    const llmMaxTokensInput = document.getElementById('llm_max_tokens_input');

    // --- State Variables ---
    let audioContext = null;
    let audioBufferQueue = [];
    let isPlayingAudio = false;
    let nextAudioStartTime = 0;
    let currentAudioBufferDuration = 0;
    let clientMinBufferDuration = 0.1;
    let fetchStreamReaderTTS = null;
    
    let chatHistory = []; // Model for chat content
    let currentLLMStreamController = null;
    
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    let currentTTSPlaybackResolver = null;

    let isPushToTalkActive = false;
    let spaceBarIsDown = false;

    // --- NEW State Variables for Delayed/Queued Text Display ---
    let llmDisplayQueue = [];
    let isDisplayingFromLLMQueue = false;
    let initialTextDisplayDelayMs = 950; // Delay for the first text appearance
    let subsequentChunkDisplayIntervalMs = 40; // Speed at which queued text streams out
    let firstLLMChunkReceived = false;
    let llmStreamCompleted = false; // Tracks if the LLM has finished sending all data
    let accumulatedLLMTextForDisplay = ""; // Holds the text that has been passed to the display queue processor
    let currentAssistantMessageContentElement = null; // Direct DOM reference to update

    // --- Initialization Functions ---
    function initializeTTSVoices() { /* ... (same as your baseline) ... */
        if (!ttsVoiceSelect) { console.warn("TTS Voice select dropdown not found."); return; }
        ttsVoices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice;
            option.textContent = voice.charAt(0).toUpperCase() + voice.slice(1);
            if (voice === "tara") option.selected = true;
            ttsVoiceSelect.appendChild(option);
        });
    }
    function initializeSliders(sliderConfigArray) { /* ... (same as your baseline) ... */
        sliderConfigArray.forEach(config => {
            const slider = document.getElementById(config.id);
            const valueSpan = document.getElementById(config.valueId);
            if (!slider || !valueSpan) { console.warn(`Slider/span not found for ${config.id}`); return; }
            const updateValue = () => {
                const val = parseFloat(slider.value);
                valueSpan.textContent = config.isFloat ? val.toFixed(config.precision || 1) : slider.value;
                if (config.id === 'client_buffer_duration_slider') clientMinBufferDuration = val;
            };
            slider.addEventListener('input', updateValue);
            updateValue();
        });
    }
    function initializeTabs() { /* ... (same as your baseline) ... */
        const tabContainers = document.querySelectorAll('.tab-container'); 
        if (tabContainers.length === 0) { console.warn("No elements with class 'tab-container' found for tabs."); return; }
        tabContainers.forEach(tabContainer => {
            const tabButtons = tabContainer.querySelectorAll('button[role="tab"]');
            const tabPanels = tabContainer.querySelectorAll('div[role="tabpanel"]');
            if (tabButtons.length === 0 || tabPanels.length === 0) return;
            tabButtons.forEach(tab => {
                tab.addEventListener('click', (event) => {
                    event.preventDefault(); 
                    tabButtons.forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
                    tabPanels.forEach(p => { p.classList.remove('active'); p.style.display = 'none'; });
                    tab.classList.add('active'); tab.setAttribute('aria-selected', 'true');
                    const controlledPanelId = tab.getAttribute('aria-controls');
                    const activePanel = tabContainer.querySelector(`#${controlledPanelId}`); 
                    if (activePanel) { activePanel.classList.add('active'); activePanel.style.display = 'block'; }
                });
            });
            let anActiveTabExists = false;
            tabButtons.forEach(tb => {
                if (tb.classList.contains('active') && tb.getAttribute('aria-selected') === 'true') {
                    const panelId = tb.getAttribute('aria-controls');
                    const panel = tabContainer.querySelector(`#${panelId}`);
                    if (panel) { panel.style.display = 'block'; panel.classList.add('active'); anActiveTabExists = true; }
                }
            });
            if (!anActiveTabExists && tabButtons.length > 0) tabButtons[0].click();
        });
    }

    // --- Chat History Management ---
    function renderChatHistory() {
        if (!chatHistoryDisplay) { console.warn("Chat history display element not found."); return null; }
        chatHistoryDisplay.innerHTML = '';
        let lastAssistantContentDiv = null;

        chatHistory.forEach(msg => {
            const msgDiv = document.createElement('div');
            msgDiv.classList.add('chat-message', msg.role === 'user' ? 'user-message' : 'assistant-message');
            
            const strong = document.createElement('strong');
            strong.textContent = msg.role === 'user' ? 'You:' : 'Assistant:';
            msgDiv.appendChild(strong);
            
            const contentDiv = document.createElement('div');
            if (msg.role === 'assistant' && msg.isStreaming) {
                contentDiv.classList.add('streaming-llm-content');
            }
            contentDiv.textContent = msg.content; 
            msgDiv.appendChild(contentDiv);
            
            chatHistoryDisplay.appendChild(msgDiv);

            if (msg.role === 'assistant') {
                lastAssistantContentDiv = contentDiv; // Keep track of the last one rendered
            }
        });
        chatHistoryDisplay.scrollTop = chatHistoryDisplay.scrollHeight;
        return lastAssistantContentDiv; // Return for direct manipulation if needed
    }

    function addUserMessageToChat(text) {
        chatHistory.push({ role: 'user', content: text });
        renderChatHistory();
    }

    function addAssistantMessageToChat(text, isStreaming = false) {
        let existingMessageIndex = -1;
        if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'assistant') {
            existingMessageIndex = chatHistory.length - 1;
        }

        if (isStreaming && existingMessageIndex !== -1 && chatHistory[existingMessageIndex].isStreaming) {
            chatHistory[existingMessageIndex].content = text; // Update model
        } else {
             // If last message was "..." and we are providing actual content
            if (existingMessageIndex !== -1 && chatHistory[existingMessageIndex].content === "..." && text !== "...") {
                chatHistory[existingMessageIndex].content = text;
                chatHistory[existingMessageIndex].isStreaming = isStreaming;
            } else {
                chatHistory.push({ role: 'assistant', content: text, isStreaming: isStreaming });
            }
        }
        return renderChatHistory(); // Renders and returns the last assistant content div
    }

    function finalizeAssistantMessage() {
        if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'assistant') {
            chatHistory[chatHistory.length - 1].isStreaming = false;
            // The content in chatHistory model should be the final accumulatedLLMTextForDisplay
            // This is handled by processLLMDisplayQueue when llmStreamCompleted is true and queue is empty
        }
        renderChatHistory(); // Re-render to remove streaming class, etc.
    }
    
    // --- TTS Audio Playback Functions ---
    function playNextAudioBuffer() { /* ... (same as your baseline) ... */
        if (audioBufferQueue.length === 0) {
            isPlayingAudio = false;
            if (fetchStreamReaderTTS === null && currentTTSPlaybackResolver) {
                currentTTSPlaybackResolver();
                currentTTSPlaybackResolver = null;
            }
            return;
        }
        if (isPlayingAudio) return;
        isPlayingAudio = true;
        const audioBuffer = audioBufferQueue.shift();
        currentAudioBufferDuration -= audioBuffer.duration;
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        const scheduleTime = Math.max(nextAudioStartTime, audioContext.currentTime + 0.005);
        source.start(scheduleTime);
        nextAudioStartTime = scheduleTime + audioBuffer.duration;
        source.onended = () => {
            isPlayingAudio = false;
            playNextAudioBuffer(); 
        };
    }
    async function processAudioStreamTTS(response) { /* ... (same as your baseline) ... */
        if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === 'suspended') {
            try { await audioContext.resume(); } catch (e) { console.error("AudioContext resume failed:", e); }
        }
        nextAudioStartTime = audioContext.currentTime; 
        fetchStreamReaderTTS = response.body.getReader(); 
        const sampleRate = parseInt(response.headers.get('X-Sample-Rate') || "24000", 10);
        try {
            while (true) {
                const { done, value } = await fetchStreamReaderTTS.read();
                if (done) {
                    fetchStreamReaderTTS = null; 
                    if (audioBufferQueue.length === 0 && !isPlayingAudio && currentTTSPlaybackResolver) {
                        currentTTSPlaybackResolver();
                        currentTTSPlaybackResolver = null;
                    } else if (audioBufferQueue.length > 0 && !isPlayingAudio) {
                        playNextAudioBuffer();
                    }
                    break; 
                }
                if (value) {
                    const float32Array = new Float32Array(value.buffer, value.byteOffset, value.byteLength / Float32Array.BYTES_PER_ELEMENT);
                    if (float32Array.length === 0) continue;
                    const buffer = audioContext.createBuffer(1, float32Array.length, sampleRate);
                    buffer.copyToChannel(float32Array, 0);
                    audioBufferQueue.push(buffer);
                    currentAudioBufferDuration += buffer.duration;
                    if (!isPlayingAudio && currentAudioBufferDuration >= clientMinBufferDuration) {
                        playNextAudioBuffer();
                    }
                }
            }
        } catch (error) {
            console.error("Error reading TTS audio stream:", error);
            if (fetchStreamReaderTTS) { 
                try { await fetchStreamReaderTTS.cancel("Error in TTS stream processing"); } 
                catch(e) {console.warn("Error cancelling reader on error:",e); }
            }
            fetchStreamReaderTTS = null; 
            audioBufferQueue = []; 
            currentAudioBufferDuration = 0;
            if (currentTTSPlaybackResolver) { 
                currentTTSPlaybackResolver();
                currentTTSPlaybackResolver = null;
            }
        }
    }
    async function speakText(text) { /* ... (same as your baseline) ... */
        if(audioPlayer) {
            audioPlayer.src = ''; audioPlayer.pause(); audioPlayer.removeAttribute('src');
        }
        audioBufferQueue = []; currentAudioBufferDuration = 0; isPlayingAudio = false;
        if (fetchStreamReaderTTS) { 
            try { await fetchStreamReaderTTS.cancel("New TTS request"); } 
            catch(e) { console.warn("Error cancelling previous TTS reader", e); }
            fetchStreamReaderTTS = null;
        }
        if (currentTTSPlaybackResolver) { currentTTSPlaybackResolver(); /* Resolve any lingering promise */ }
        const playbackCompletePromise = new Promise(resolve => { currentTTSPlaybackResolver = resolve; });
        const ttsPayload = {
            text: text, voice: ttsVoiceSelect.value,
            tts_temperature: parseFloat(document.getElementById('tts_temp_slider').value),
            tts_top_p: parseFloat(document.getElementById('tts_top_p_slider').value),
            tts_repetition_penalty: parseFloat(document.getElementById('tts_rep_penalty_slider').value),
            buffer_groups: parseInt(document.getElementById('tts_buffer_groups_slider').value, 10),
            padding_ms: parseInt(document.getElementById('tts_padding_ms_slider').value, 10),
            min_decode_batch_groups: parseInt(document.getElementById('tts_batch_groups_slider').value, 10)
        };
        try {
            const response = await fetch('/api/tts/stream', { 
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ttsPayload) 
            });
            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: `TTS API Error: ${response.status}` }));
                throw new Error(errData.detail || `TTS API Error: ${response.status}`);
            }
            await processAudioStreamTTS(response); 
            await playbackCompletePromise;
        } catch (error) {
            console.error('TTS Request failed:', error);
            addAssistantMessageToChat(`(TTS Error: ${error.message})`); // This will call renderChatHistory
            if (currentTTSPlaybackResolver) { currentTTSPlaybackResolver(); currentTTSPlaybackResolver = null; }
        }
    }

    // --- LLM Interaction Functions ---
    // New helper function to process the display queue
    function processLLMDisplayQueue() {
        if (llmDisplayQueue.length > 0) {
            isDisplayingFromLLMQueue = true;
            const chunkToDisplay = llmDisplayQueue.shift();
            accumulatedLLMTextForDisplay += chunkToDisplay;
            
            if (currentAssistantMessageContentElement) {
                currentAssistantMessageContentElement.textContent = accumulatedLLMTextForDisplay;
                if(chatHistoryDisplay) chatHistoryDisplay.scrollTop = chatHistoryDisplay.scrollHeight;
            } else {
                // Fallback: This might cause rapid re-renders if currentAssistantMessageContentElement is not found
                addAssistantMessageToChat(accumulatedLLMTextForDisplay, true);
            }
            setTimeout(processLLMDisplayQueue, subsequentChunkDisplayIntervalMs);
        } else {
            isDisplayingFromLLMQueue = false;
            if (llmStreamCompleted) { // LLM stream done AND display queue empty
                // Update the model in chatHistory with the final accumulated text
                if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'assistant') {
                    chatHistory[chatHistory.length - 1].content = accumulatedLLMTextForDisplay;
                }
                finalizeAssistantMessage(); // Finalizes isStreaming flag and re-renders
            }
        }
    }

    async function fetchLLMResponse(promptText) {
        // Reset state for this new response
        llmDisplayQueue = [];
        isDisplayingFromLLMQueue = false;
        firstLLMChunkReceived = false;
        llmStreamCompleted = false;
        accumulatedLLMTextForDisplay = "";

        // Create the initial assistant message bubble and get its content element for direct updates
        currentAssistantMessageContentElement = addAssistantMessageToChat("...", true); 
        if (currentAssistantMessageContentElement && currentAssistantMessageContentElement.textContent === "...") {
             currentAssistantMessageContentElement.textContent = ""; // Clear "..." for streaming
        }


        const historyForAPI = chatHistory.slice(0, -2) // Exclude current user prompt and assistant placeholder
                              .filter(msg => msg.content !== "...") 
                              .map(msg => ({ role: msg.role, content: msg.content })); 

        const llmPayload = {
            prompt: promptText, 
            history: historyForAPI,
            temperature: parseFloat(document.getElementById('llm_temp_slider').value),
            top_p: parseFloat(document.getElementById('llm_top_p_slider').value),
            max_tokens: parseInt(llmMaxTokensInput.value, 10),
            repetition_penalty: parseFloat(document.getElementById('llm_rep_penalty_slider').value),
            top_k: parseInt(document.getElementById('llm_top_k_slider').value, 10)
        };
        if (llmPayload.top_k === 0) delete llmPayload.top_k;

        let fullLLMResponse = ""; // For the final TTS call
        currentLLMStreamController = new AbortController();

        try {
            const response = await fetch('/api/llm/chat/stream', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify(llmPayload), 
                signal: currentLLMStreamController.signal 
            });

            if (!response.ok) {
                let errorDetailMessage = `LLM API Error: Status ${response.status} - ${response.statusText}`;
                try { 
                    const errorData = await response.json(); 
                    if (errorData && errorData.detail) { errorDetailMessage = errorData.detail; }
                    else if (errorData && errorData.error && errorData.error.message) { errorDetailMessage = errorData.error.message; }
                    else if (typeof errorData === 'string') {errorDetailMessage = errorData;}
                    else { errorDetailMessage = JSON.stringify(errorData); }
                } catch (e) { console.warn("Could not parse LLM error JSON:", e); }
                throw new Error(errorDetailMessage);
            }

            const reader = response.body.getReader(); 
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) { 
                    llmStreamCompleted = true;
                    // If display queue is already empty (e.g. very short response or fast display)
                    if (!isDisplayingFromLLMQueue && llmDisplayQueue.length === 0) {
                        if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'assistant') {
                           chatHistory[chatHistory.length - 1].content = accumulatedLLMTextForDisplay;
                        }
                        finalizeAssistantMessage();
                    }
                    console.log("LLM stream finished."); 
                    break; 
                }
                
                const chunk = decoder.decode(value, { stream: true });
                fullLLMResponse += chunk;    // Accumulate for TTS
                llmDisplayQueue.push(chunk); // Add to display queue

                if (!firstLLMChunkReceived) {
                    firstLLMChunkReceived = true;
                    setTimeout(() => {
                        if (!isDisplayingFromLLMQueue) { // Start processing only if not already doing so
                            processLLMDisplayQueue();
                        }
                    }, initialTextDisplayDelayMs);
                } else {
                    // If initial delay already passed and queue processor isn't running (e.g. very fast chunks after initial delay)
                    // ensure it starts if displayQueue was empty and a new chunk arrived.
                    // This is implicitly handled by processLLMDisplayQueue calling itself with setTimeout.
                }
            }
            // TTS call remains after the entire LLM response is received and processed for display.
            // The 'finalizeAssistantMessage' call inside processLLMDisplayQueue (when llmStreamCompleted is true)
            // ensures the text model (chatHistory) is up-to-date before TTS might be called.
            
            const currentMode = modeSelect.value;
            if (currentMode === 'llm_tts' && fullLLMResponse.trim() && !fullLLMResponse.trim().startsWith("[Error")) {
                if (generateButton) generateButton.textContent = 'Synthesizing...'; 
                await speakText(fullLLMResponse.trim());
            }

        } catch (error) {
            console.error('LLM Request failed:', error);
            llmStreamCompleted = true; // Critical for finalization path
            const errorMsg = error.name === 'AbortError' ? "(LLM request cancelled by user)" : `(LLM Error: ${error.message})`;
            
            // Update the DOM directly if possible, then update the model
            if (currentAssistantMessageContentElement) {
                currentAssistantMessageContentElement.textContent = errorMsg;
                currentAssistantMessageContentElement.classList.remove('streaming-llm-content');
            }
            // Update the chatHistory model
            if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'assistant') {
                chatHistory[chatHistory.length - 1].content = errorMsg;
            } else { // Should not happen if placeholder was added, but as a fallback
                addAssistantMessageToChat(errorMsg, false);
            }
            finalizeAssistantMessage(); // Finalize streaming state in model
        } finally { 
            currentLLMStreamController = null; 
            // Button re-enablement is handled by the main appForm submit listener's finally block
        }
    }

    // --- STT Functions ---
    const SVG_MIC_ICON = `
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" style="width:1.3em; height:1.3em;">
            <path d="M12 18.75a6 6 0 0 0 6-6v-1.5a6 6 0 0 0-12 0v1.5a6 6 0 0 0 6 6Z" />
            <path d="M12 22.5a3 3 0 0 1-3-3v-1.5a3 3 0 0 1 6 0v1.5a3 3 0 0 1-3 3Z" />
            <path d="M8.25 12a3.75 3.75 0 0 0-3.75 3.75v.75a.75.75 0 0 0 1.5 0v-.75a2.25 2.25 0 0 1 2.25-2.25H12v-.75A3.75 3.75 0 0 0 8.25 12Z" />
            <path d="M12 12h3.75a2.25 2.25 0 0 1 2.25 2.25v.75a.75.75 0 0 0 1.5 0v-.75A3.75 3.75 0 0 0 15.75 12H12Z" />
        </svg>`;

    async function startRecording() { /* ... (same as your baseline) ... */
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Browser doesn't support audio recording.");
            if (recordStatus) recordStatus.textContent = "Not supported."; return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const options = { mimeType: 'audio/webm;codecs=opus' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) { delete options.mimeType; }
            mediaRecorder = new MediaRecorder(stream, options); audioChunks = [];
            mediaRecorder.ondataavailable = event => { audioChunks.push(event.data); };
            mediaRecorder.onstop = async () => {
                let fileExtension = ".webm"; let blobType = mediaRecorder.mimeType || 'audio/webm';
                if (mediaRecorder && mediaRecorder.mimeType) {
                    if (mediaRecorder.mimeType.includes("audio/wav")) fileExtension = ".wav";
                    else if (mediaRecorder.mimeType.includes("audio/mp3")) fileExtension = ".mp3";
                    else if (mediaRecorder.mimeType.includes("audio/ogg")) fileExtension = ".ogg";
                }
                const audioBlob = new Blob(audioChunks, { type: blobType }); audioChunks = [];
                stream.getTracks().forEach(track => track.stop());
                await sendAudioForTranscription(audioBlob, `user_recording${fileExtension}`);
            };
            mediaRecorder.start(); isRecording = true;
            if (recordButton) { recordButton.innerHTML = '🛑'; recordButton.classList.add('recording'); }
            if (recordStatus) recordStatus.textContent = "Recording...";
        } catch (err) {
            console.error("Mic error:", err); alert("Mic error. Check permissions.");
            if (recordStatus) recordStatus.textContent = "Mic error!"; isRecording = false;
            if (recordButton) { recordButton.innerHTML = SVG_MIC_ICON; recordButton.classList.remove('recording'); }
        }
    }
    function stopRecording() { /* ... (same as your baseline) ... */
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop(); isRecording = false;
            if (recordButton) { recordButton.innerHTML = SVG_MIC_ICON; recordButton.classList.remove('recording'); }
            if (recordStatus) recordStatus.textContent = "Processing...";
        }
    }
    async function sendAudioForTranscription(audioBlob, fileName) { /* ... (same as your baseline, PTT logic included) ... */
        const formData = new FormData(); formData.append("audio_file", audioBlob, fileName);
        if (recordStatus) recordStatus.textContent = "Transcribing...";
        if (generateButton) generateButton.disabled = true;
        let transcriptionSuccessful = false;
        try {
            const response = await fetch('/api/stt/transcribe', { method: 'POST', body: formData });
            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: `STT API Error: ${response.status}` }));
                throw new Error(err.detail || `STT Error: ${response.status}`);
            }
            const result = await response.json();
            if (result.error) { throw new Error(result.error); }
            if (textInput) textInput.value = result.text;
            if (recordStatus) recordStatus.textContent = "Transcribed!";
            transcriptionSuccessful = true;
        } catch (error) {
            console.error('STT Request failed:', error); alert(`Transcription error: ${error.message}`);
            if (recordStatus) recordStatus.textContent = "STT failed!";
        } finally {
            if (generateButton) generateButton.disabled = false;
            if (transcriptionSuccessful && isPushToTalkActive) {
                if (textInput && textInput.value.trim() !== "") {
                    if (generateButton) generateButton.click();
                } else { console.log("PTT: No transcribed text to submit.");}
            }
            if(isPushToTalkActive) isPushToTalkActive = false;
            spaceBarIsDown = false; 
        }
    }

    if (recordButton) { /* ... (same as your baseline) ... */
        recordButton.addEventListener('click', () => {
            if (isRecording) { stopRecording(); } else { startRecording(); }
        });
    }

    // --- Push-to-Talk Event Listeners --- (same as your baseline)
    document.addEventListener('keydown', async (event) => { /* ... (same as your baseline) ... */
        if (event.code === 'Space') {
            const activeEl = document.activeElement;
            if (activeEl && (activeEl.tagName.toLowerCase() === 'input' || activeEl.tagName.toLowerCase() === 'textarea')) {
                if (activeEl === textInput && textInput.value.length > 0) return;
                if (activeEl !== textInput) return; 
            }
            event.preventDefault(); 
            if (!isRecording && !spaceBarIsDown) {
                spaceBarIsDown = true; isPushToTalkActive = true; 
                await startRecording();
            }
        }
    });
    document.addEventListener('keyup', async (event) => { /* ... (same as your baseline) ... */
        if (event.code === 'Space') {
            const activeEl = document.activeElement;
            if (activeEl === textInput && !isPushToTalkActive && !spaceBarIsDown) return;
            if (activeEl && (activeEl.tagName.toLowerCase() === 'input' || activeEl.tagName.toLowerCase() === 'textarea') && activeEl !== textInput) return;
            if (spaceBarIsDown) {
                spaceBarIsDown = false; 
                if (isRecording && isPushToTalkActive) { 
                    await stopRecording(); 
                } else if (isPushToTalkActive) {
                    isPushToTalkActive = false; // Reset if recording didn't actually start
                }
            }
        }
    });

    // --- Main Form Submit Handler ---
    if (appForm) { /* ... (same as your baseline, with PTT check for empty userText) ... */
        appForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const currentMode = modeSelect.value;
            const userText = textInput.value.trim();
            if (!userText && !isPushToTalkActive) { alert("Please enter text or record audio first."); return; }
            if (!userText && isPushToTalkActive) { /* PTT submitted empty */ return; }
            if (generateButton) { generateButton.disabled = true; generateButton.textContent = 'Processing...'; }
            addUserMessageToChat(userText);
            if (currentMode === 'tts_only' && textInput) textInput.value = ''; 
            try {
                if (currentMode === 'tts_only') {
                    if (generateButton) generateButton.textContent = 'Synthesizing...';
                    await speakText(userText);
                } else if (currentMode === 'llm_only' || currentMode === 'llm_tts') {
                    if (generateButton) generateButton.textContent = 'Thinking...';
                    await fetchLLMResponse(userText); 
                }
            } catch (e) {
                console.error("Error in main submission process:", e);
                addAssistantMessageToChat(`(App Error: ${e.message})`);
                finalizeAssistantMessage(); // Ensure finalization
            } finally {
                if (generateButton) { generateButton.disabled = false; generateButton.textContent = 'Generate'; }
                if ((currentMode === 'llm_only' || currentMode === 'llm_tts') && textInput) textInput.value = '';
            }
        });
    } else { console.warn("App form not found."); }

    // --- Initial Page Setup Calls ---
    initializeTTSVoices();
    initializeSliders(ttsSliders);
    initializeSliders(llmSliders);
    if (document.getElementById('client_buffer_duration_slider')) {
        clientMinBufferDuration = parseFloat(document.getElementById('client_buffer_duration_slider').value);
    }
    initializeTabs();
    renderChatHistory();
    if (recordButton) recordButton.innerHTML = SVG_MIC_ICON;

}); // End of DOMContentLoaded