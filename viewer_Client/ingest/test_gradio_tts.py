import gradio as gr
import numpy as np
import sounddevice as sd
import sys
import os
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ingest.audio_pipeline import AudioPipeline
from src.ingest.tts_adapter import TTSAdapter

def process_and_stream(audio_data):
    """
    Takes audio data from Gradio (sr, data), processes it via TTS Adapter,
    and streams it to speakers.
    """
    if audio_data is None:
        return "No audio provided."

    start_time = time.time()
    sr, data = audio_data
    
    print(f"Received audio: Sample Rate={sr}, Shape={data.shape}, Dtype={data.dtype}")

    # Initialize Pipeline
    pipeline = AudioPipeline()
    adapter = TTSAdapter(pipeline)
    adapter.start_stream("gradio_test_stream")

    # Determine channels
    channels = 1
    if data.ndim > 1:
        channels = data.shape[1]
    
    # Ensure data is consistent (Gradio usually provides int16 or float32)
    # Adapter expects bytes.
    # Note: If data is int16, we should pass that dtype to adapter or convert to float32 first.
    # Let's inspect dtype provided/expected. 
    # The adapter defaults to float32 in current implementation but we passed dtype to np.frombuffer.
    # If we pass raw bytes of int16, we must tell adapter it's int16.
    
    # Let's normalize to float32 before sending to facilitate:
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
        
    # Now data is float32.
    adapter_dtype = np.float32
    
    # Feed data to adapter
    print("Feeding audio to adapter...")
    adapter.on_audio_data(data.tobytes(), sr, channels=channels, dtype=adapter_dtype)
    adapter.stop_stream(drain=True)
    
    # Stream playback
    # Pipeline outputs 16kHz mono float32
    output_sr = 16000
    print("Starting playback stream...")
    
    try:
        # Create an output stream
        with sd.OutputStream(samplerate=output_sr, channels=1, dtype='float32') as stream:
            chunk_count = 0
            for window in adapter.consume_windows():
                if chunk_count == 0:
                    latency_ms = (time.time() - start_time) * 1000
                    print(f"Time to first chunk (Latency): {latency_ms:.2f} ms")
                    
                # Write to audio device
                start = window.center_start
                end = start + window.center_num_samples
                stream.write(window.pcm[start:end])
                chunk_count += 1
                
        return f"Streaming complete. Processed {chunk_count} chunks."
    except Exception as e:
        return f"Error during playback: {e}"

# Build Gradio Interface
description = """
Upload an audio file. It will be:
1. Sent to TTS Stream Adapter.
2. Resampled to 16kHz Mono.
3. Chunked to 20ms.
4. Streamed directly to your speakers.
"""

iface = gr.Interface(
    fn=process_and_stream,
    inputs=gr.Audio(type="numpy", label="Input Audio"),
    outputs="text",
    title="TTS Stream Adapter Test",
    description=description
)

if __name__ == "__main__":
    # Launch on localhost
    print("Launching Gradio interface...")
    iface.launch(server_name="127.0.0.1")
