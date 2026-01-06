from src.ingest.audio_pipeline import AudioPipeline
import numpy as np

class TTSAdapter:
    def __init__(self, audio_pipeline: AudioPipeline):
        self.pipeline = audio_pipeline
        self.is_active = False
        self.current_stream_id = None
        self._draining = False
        
    def start_stream(self, stream_id: str):
        self.current_stream_id = stream_id
        self.is_active = True
        self._draining = False
        self.pipeline.flush()
        # TODO: Handle start events
        
    def stop_stream(self, drain: bool = True):
        self.is_active = False
        if drain:
            self.pipeline.end_stream(extra_hops=1)
            self._draining = True
        else:
            self._draining = False
            self.current_stream_id = None
        
    def on_audio_data(self, audio_bytes: bytes, sample_rate: int, channels: int = 1, dtype=np.float32):
        """
        Callback/Method to be called when new audio data arrives from TTS.
        """
        if not self.is_active:
            return
            
        # Convert bytes to numpy array
        pcm_data = np.frombuffer(audio_bytes, dtype=dtype)
        
        # Reshape if multiple channels
        if channels > 1:
            # Check if length is divisible by channels
            if len(pcm_data) % channels != 0:
                # Log warning or handle partial frames? For now, just truncate or error.
                # Truncating to safe length
                safe_len = (len(pcm_data) // channels) * channels
                pcm_data = pcm_data[:safe_len]
                
            pcm_data = pcm_data.reshape(-1, channels)
        
        # Feed to pipeline
        self.pipeline.add_audio(pcm_data, sample_rate)
        
    def consume_windows(self):
        """
        Generator yielding available windows from the pipeline.
        """
        if not self.current_stream_id:
            return
            
        while True:
            window = self.pipeline.get_next_window(self.current_stream_id)
            if window:
                yield window
            else:
                if self._draining:
                    self._draining = False
                    self.current_stream_id = None
                break

    def consume_chunks(self):
        return self.consume_windows()
