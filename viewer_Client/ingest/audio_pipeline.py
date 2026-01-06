from __future__ import annotations

from dataclasses import dataclass
import io
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile


@dataclass
class AudioChunk:
    stream_id: str
    seq: int
    pcm: np.ndarray
    sample_rate: int
    window_num_samples: int
    hop_num_samples: int
    center_start: int
    center_num_samples: int
    window_time_ns: int
    center_time_ns: int
    audio_time_ns: int


def _to_mono_float32(pcm: np.ndarray) -> np.ndarray:
    if pcm.size == 0:
        return np.empty((0,), dtype=np.float32)

    if pcm.ndim > 1:
        pcm = np.mean(pcm, axis=1, dtype=np.float32)

    if np.issubdtype(pcm.dtype, np.floating):
        return pcm.astype(np.float32, copy=False)

    if pcm.dtype == np.int16:
        return (pcm.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    if pcm.dtype == np.int32:
        return (pcm.astype(np.float32) / 2147483648.0).astype(np.float32, copy=False)
    if pcm.dtype == np.uint8:
        return ((pcm.astype(np.float32) - 128.0) / 128.0).astype(np.float32, copy=False)

    return pcm.astype(np.float32)


def _resample_mono_float32(pcm: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if original_sample_rate == target_sample_rate:
        return pcm.astype(np.float32, copy=False)

    gcd = math.gcd(original_sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = original_sample_rate // gcd

    out = signal.resample_poly(pcm, up, down)
    return out.astype(np.float32, copy=False)


def _float32_to_pcm16le(pcm: np.ndarray) -> np.ndarray:
    if pcm.size == 0:
        return np.empty((0,), dtype="<i2")

    clipped = np.clip(pcm, -1.0, 1.0).astype(np.float32, copy=False)
    scaled = np.round(clipped * 32768.0)
    scaled = np.clip(scaled, -32768.0, 32767.0)
    return scaled.astype("<i2")


def process_wav_bytes_to_pcm16_chunks(
    wav_bytes: bytes,
    target_sample_rate: int = 16000,
    chunk_duration_s: float = 1.0,
) -> Tuple[int, List[Tuple[int, bytes]]]:
    """
    Decode a WAV file (bytes), resample to `target_sample_rate`, convert to mono PCM16,
    and return sequential non-overlapping chunks.

    Returns: (sample_rate, [(startSampleIndex, pcm16le_bytes), ...])
    """
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be > 0")

    src_sample_rate, pcm = wavfile.read(io.BytesIO(wav_bytes))
    pcm_f32 = _to_mono_float32(pcm)
    pcm_f32 = _resample_mono_float32(pcm_f32, int(src_sample_rate), int(target_sample_rate))

    chunk_samples = int(round(target_sample_rate * chunk_duration_s))
    if chunk_samples <= 0:
        raise ValueError("Invalid chunk size")

    chunks: List[Tuple[int, bytes]] = []
    start = 0
    while start < pcm_f32.shape[0]:
        end = min(pcm_f32.shape[0], start + chunk_samples)
        pcm16 = _float32_to_pcm16le(pcm_f32[start:end])
        chunks.append((start, pcm16.tobytes()))
        start += chunk_samples

    return int(target_sample_rate), chunks

class AudioPipeline:
    def __init__(self, target_sample_rate=16000, window_duration_s=1.0, hop_duration_s=0.5):
        self.target_sample_rate = target_sample_rate
        self.window_num_samples = int(round(target_sample_rate * window_duration_s))
        self.hop_num_samples = int(round(target_sample_rate * hop_duration_s))
        self.center_start = (self.window_num_samples - self.hop_num_samples) // 2
        self.center_num_samples = self.hop_num_samples

        if self.window_num_samples <= 0:
            raise ValueError("window_duration_s must be > 0")
        if self.hop_num_samples <= 0 or self.hop_num_samples > self.window_num_samples:
            raise ValueError("hop_duration_s must be > 0 and <= window_duration_s")
        if self.center_start < 0 or self.center_start + self.center_num_samples > self.window_num_samples:
            raise ValueError("Invalid center region for window/hop configuration")

        self._buffer = np.empty((0,), dtype=np.float32)
        self._start = 0
        self._ended = False

        # Sequence number for windows
        self.seq_counter = 0
        
        self._next_window_start_sample_index = 0
        
    def add_audio(self, pcm_data: np.ndarray, original_sample_rate: int):
        """
        Ingest audio data, resample to target rate, and append to buffer.
        Assumes pcm_data is numpy array. If stereo, converts to mono.
        """
        # 1. Convert to mono if necessary
        if pcm_data.ndim > 1 and pcm_data.shape[1] > 1:
            pcm_data = np.mean(pcm_data, axis=1, dtype=np.float32)
            
        # 2. Resample if necessary
        if original_sample_rate != self.target_sample_rate:
            gcd = math.gcd(original_sample_rate, self.target_sample_rate)
            up = self.target_sample_rate // gcd
            down = original_sample_rate // gcd
            pcm_data = signal.resample_poly(pcm_data, up, down)
            if pcm_data.dtype != np.float32:
                 pcm_data = pcm_data.astype(np.float32)
            
        # 3. Ensure float32 for consistency (or int16 based on needs) - lets stick to float32 normalized -1.0 to 1.0 for now if input is such
        # For simplicity, assuming input is already reasonable type, just ensuring shape
        
        # Append to buffer
        if len(pcm_data) > 0:
            if pcm_data.dtype != np.float32:
                pcm_data = pcm_data.astype(np.float32)
            self._buffer = np.concatenate([self._buffer, pcm_data])
        
    def get_next_window(self, stream_id: str) -> Optional[AudioChunk]:
        """
        Extracts the next 1.0s window from the buffer if available.
        Returns None if not enough data.
        """
        available = len(self._buffer) - self._start
        if available < self.window_num_samples:
            return None
            
        window_start = self._start
        window_end = window_start + self.window_num_samples
        window_pcm = self._buffer[window_start:window_end].copy()

        window_time_ns = int((self._next_window_start_sample_index / self.target_sample_rate) * 1e9)
        center_time_ns = int(((self._next_window_start_sample_index + self.center_start) / self.target_sample_rate) * 1e9)

        chunk = AudioChunk(
            stream_id=stream_id,
            seq=self.seq_counter,
            pcm=window_pcm,
            sample_rate=self.target_sample_rate,
            window_num_samples=self.window_num_samples,
            hop_num_samples=self.hop_num_samples,
            center_start=self.center_start,
            center_num_samples=self.center_num_samples,
            window_time_ns=window_time_ns,
            center_time_ns=center_time_ns,
            audio_time_ns=center_time_ns,
        )
        
        # Update counters
        self.seq_counter += 1
        self._start += self.hop_num_samples
        self._next_window_start_sample_index += self.hop_num_samples

        if self._start >= self.window_num_samples * 4:
            self._buffer = self._buffer[self._start:]
            self._start = 0
        
        return chunk
        
    def flush(self):
        """Clear buffer and reset counters"""
        self._buffer = np.empty((0,), dtype=np.float32)
        self._start = 0
        self._ended = False
        self.seq_counter = 0
        self._next_window_start_sample_index = 0

    def end_stream(self, extra_hops: int = 1):
        if self._ended:
            return
        self._ended = True

        if extra_hops < 0:
            raise ValueError("extra_hops must be >= 0")

        available = len(self._buffer) - self._start
        if available <= 0:
            return

        over = max(0, available - self.window_num_samples)
        hops_to_cover = (over + self.hop_num_samples - 1) // self.hop_num_samples
        total_hops = hops_to_cover + extra_hops

        required = self.window_num_samples + total_hops * self.hop_num_samples
        pad_len = required - available
        if pad_len <= 0:
            return

        self._buffer = np.concatenate([self._buffer, np.zeros((pad_len,), dtype=np.float32)])

    def get_buffer_fill_ms(self):
        return ((len(self._buffer) - self._start) / self.target_sample_rate) * 1000.0
