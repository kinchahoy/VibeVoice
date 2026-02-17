#!/usr/bin/env python
"""
VibeVoice ASR Batch Inference Demo Script

This script supports batch inference for ASR model and compares results
between batch processing and single-sample processing.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import json
import re
from typing import List, Dict, Any, Optional
from functools import wraps
import tempfile
import threading

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse


class VibeVoiceASRBatchInference:
    """Batch inference wrapper for VibeVoice ASR model."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda", 
        dtype: torch.dtype = torch.float16,
        attn_implementation: str = "sdpa"
    ):
        """
        Initialize the ASR batch inference pipeline.
        
        Args:
            model_path: Path to the pretrained model
            device: Device to run inference on (cuda, mps, xpu, cpu, auto)
            dtype: Data type for model weights (always forced to float16)
            attn_implementation: Attention implementation to use ('flash_attention_2', 'sdpa', 'eager')
        """
        print(f"Loading VibeVoice ASR model from {model_path}")
        
        # Load processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )
        
        # Load model with specified attention implementation.
        # Force float16 for this script to avoid accidental bfloat16 usage.
        forced_dtype = torch.float16
        if dtype != forced_dtype:
            print(f"Overriding requested dtype {dtype} -> {forced_dtype}")

        print(f"Using attention implementation: {attn_implementation}")
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=forced_dtype,
            device_map=device if device == "auto" else None,
            attn_implementation=attn_implementation,
            trust_remote_code=True
        )
        
        if device != "auto":
            self.model = self.model.to(device)
        
        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.dtype = forced_dtype
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def _prepare_generation_config(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> dict:
        """Prepare generation configuration."""
        config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # Beam search vs sampling
        if num_beams > 1:
            config["num_beams"] = num_beams
            config["do_sample"] = False  # Beam search doesn't use sampling
        else:
            config["do_sample"] = do_sample
            # Only set temperature and top_p when sampling is enabled
            if do_sample:
                config["temperature"] = temperature
                config["top_p"] = top_p
        
        return config
    
    def transcribe_batch(
        self,
        audio_inputs: List,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        context_info: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files/arrays in a single batch.
        
        Args:
            audio_inputs: List of audio file paths or (array, sampling_rate) tuples
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling
            
        Returns:
            List of transcription results
        """
        if len(audio_inputs) == 0:
            return []
        
        batch_size = len(audio_inputs)
        print(f"\nProcessing batch of {batch_size} audio(s)...")
        
        # Process all audio together
        inputs = self.processor(
            audio=audio_inputs,
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            context_info=context_info,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Print batch info
        print(f"  Input IDs shape: {inputs['input_ids'].shape}")
        print(f"  Speech tensors shape: {inputs['speech_tensors'].shape}")
        print(f"  Attention mask shape: {inputs['attention_mask'].shape}")
        
        # Generate
        generation_config = self._prepare_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode outputs for each sample in the batch
        results = []
        input_length = inputs['input_ids'].shape[1]
        
        for i, audio_input in enumerate(audio_inputs):
            # Get generated tokens for this sample (excluding input tokens)
            generated_ids = output_ids[i, input_length:]
            
            # Remove padding tokens from the end
            # Find the first eos_token or pad_token
            eos_positions = (generated_ids == self.processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[:eos_positions[0] + 1]
            
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Parse structured output
            try:
                transcription_segments = self.processor.post_process_transcription(generated_text)
            except Exception as e:
                print(f"Warning: Failed to parse structured output: {e}")
                transcription_segments = []
            
            # Get file name based on input type
            if isinstance(audio_input, str):
                file_name = audio_input
            elif isinstance(audio_input, dict) and 'id' in audio_input:
                file_name = audio_input['id']
            else:
                file_name = f"audio_{i}"
            
            results.append({
                "file": file_name,
                "raw_text": generated_text,
                "segments": transcription_segments,
                "generation_time": generation_time / batch_size,
            })
        
        print(f"  Total generation time: {generation_time:.2f}s")
        print(f"  Average time per sample: {generation_time/batch_size:.2f}s")
        
        return results
    
    def transcribe_with_batching(
        self,
        audio_inputs: List,
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files/arrays with automatic batching.
        
        Args:
            audio_inputs: List of audio file paths or (array, sampling_rate) tuples
            batch_size: Number of samples per batch
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            do_sample: Whether to use sampling
            
        Returns:
            List of transcription results
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(audio_inputs), batch_size):
            batch_inputs = audio_inputs[i:i + batch_size]
            print(f"\n{'='*60}")
            print(f"Processing batch {i//batch_size + 1}/{(len(audio_inputs) + batch_size - 1)//batch_size}")
            
            batch_results = self.transcribe_batch(
                batch_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
            )
            all_results.extend(batch_results)
        
        return all_results


def print_result(result: Dict[str, Any]):
    """Pretty print a single transcription result."""
    print(f"\nFile: {result['file']}")
    print(f"Generation Time: {result['generation_time']:.2f}s")
    print(f"\n--- Raw Output ---")
    print(result['raw_text'][:500] + "..." if len(result['raw_text']) > 500 else result['raw_text'])
    
    if result['segments']:
        print(f"\n--- Structured Output ({len(result['segments'])} segments) ---")
        for seg in result['segments'][:50]:  # Show first 50 segments
            print(f"[{seg.get('start_time', 'N/A')} - {seg.get('end_time', 'N/A')}] "
                  f"Speaker {seg.get('speaker_id', 'N/A')}: {seg.get('text', '')}...")
        if len(result['segments']) > 50:
            print(f"  ... and {len(result['segments']) - 50} more segments")


def load_dataset_and_concatenate(
    dataset_name: str,
    split: str,
    max_duration: float,
    num_audios: int,
    target_sr: int = 24000
) -> Optional[List[np.ndarray]]:
    """
    Load a HuggingFace dataset and concatenate audio samples into long audio chunks.
    (Note, just for demo purpose, not for benchmark evaluation)
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'openslr/librispeech_asr')
        split: Dataset split to use (e.g., 'test', 'test.other')
        max_duration: Maximum duration in seconds for each concatenated audio
        num_audios: Number of concatenated audios to create
        target_sr: Target sample rate (default: 24000)
    
    Returns:
        List of concatenated audio arrays, or None if loading fails
    """
    try:
        from datasets import load_dataset
        import torchcodec # just for decode audio in datasets
    except ImportError:
        print("Please install it with: pip install datasets torchcodec")
        return None        
    
    print(f"\nLoading dataset: {dataset_name} (split: {split})")
    print(f"Will create {num_audios} concatenated audio(s), each up to {max_duration:.1f}s ({max_duration/3600:.2f} hours)")
    
    try:
        # Use streaming to avoid downloading the entire dataset
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        print(f"Dataset loaded in streaming mode")
        
        concatenated_audios = []  # List of concatenated audio metadata
        
        # Create multiple concatenated audios based on num_audios
        current_chunks = []
        current_duration = 0.0
        current_samples_used = 0
        sample_idx = 0
        
        for sample in dataset:
            if len(concatenated_audios) >= num_audios:
                break
                
            if 'audio' not in sample:
                continue
            
            audio_data = sample['audio']
            audio_array = audio_data['array']
            sr = audio_data['sampling_rate']
            
            # Resample if needed
            if sr != target_sr:
                duration = len(audio_array) / sr
                new_length = int(duration * target_sr)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array) - 1, new_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
            
            chunk_duration = len(audio_array) / target_sr
            
            # Check if adding this chunk exceeds max_duration
            if current_duration + chunk_duration > max_duration:
                remaining_duration = max_duration - current_duration
                if remaining_duration > 0.5:  # Only add if > 0.5s remaining
                    samples_to_take = int(remaining_duration * target_sr)
                    current_chunks.append(audio_array[:samples_to_take])
                    current_duration += remaining_duration
                    current_samples_used += 1
                
                # Save current concatenated audio and start a new one
                if current_chunks:
                    concatenated_audios.append({
                        'array': np.concatenate(current_chunks),
                        'duration': current_duration,
                        'samples_used': current_samples_used,
                    })
                    print(f"  Created audio {len(concatenated_audios)}: {current_duration:.1f}s from {current_samples_used} samples")
                
                # Reset for next concatenated audio
                current_chunks = []
                current_duration = 0.0
                current_samples_used = 0
                
                if len(concatenated_audios) >= num_audios:
                    break
            
            current_chunks.append(audio_array)
            current_duration += chunk_duration
            current_samples_used += 1
            
            sample_idx += 1
            if sample_idx % 100 == 0:
                print(f"  Processed {sample_idx} samples...")
        
        # Don't forget the last batch if it has content
        if current_chunks and len(concatenated_audios) < num_audios:
            concatenated_audios.append({
                'array': np.concatenate(current_chunks),
                'duration': current_duration,
                'samples_used': current_samples_used,
            })
            print(f"  Created audio {len(concatenated_audios)}: {current_duration:.1f}s from {current_samples_used} samples")
        
        if not concatenated_audios:
            print("Warning: No audio samples found in dataset")
            return None
        
        # Extract arrays and print summary
        result = [a['array'] for a in concatenated_audios]
        total_duration = sum(a['duration'] for a in concatenated_audios)
        total_samples = sum(a['samples_used'] for a in concatenated_audios)
        print(f"\nCreated {len(result)} concatenated audio(s), total {total_duration:.1f}s ({total_duration/60:.1f} min) from {total_samples} samples")
        
        return result
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


_TIMESTAMP_PATTERN = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")
_SUPPORTED_AUDIO_RESPONSE_FORMATS = {
    "json",
    "text",
    "srt",
    "verbose_json",
    "vtt",
    "diarized_json",
}


def _allowed_formats_for_model(model_name: str) -> set[str]:
    """Apply model-specific response-format constraints where required."""
    name = (model_name or "").strip().lower()
    if name in {"gpt-4o-transcribe", "gpt-4o-mini-transcribe"}:
        return {"json"}
    if name == "gpt-4o-transcribe-diarize":
        return {"json", "text", "diarized_json"}
    return set(_SUPPORTED_AUDIO_RESPONSE_FORMATS)


def _timestamp_to_seconds(value: Any) -> Optional[float]:
    """Convert model timestamp values to seconds (best effort)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    match = _TIMESTAMP_PATTERN.match(text)
    if not match:
        return None

    hours_raw, minutes_raw, seconds_raw = match.groups()
    hours = int(hours_raw) if hours_raw else 0
    minutes = int(minutes_raw)
    seconds = float(seconds_raw)
    return hours * 3600 + minutes * 60 + seconds


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as WebVTT timestamp: HH:MM:SS.mmm."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def _segments_to_text(segments: List[Dict[str, Any]], raw_text: str) -> str:
    """Build plain transcript text from structured segments when possible."""
    if segments:
        text_parts = []
        for seg in segments:
            seg_text = seg.get("text", "")
            if seg_text:
                text_parts.append(str(seg_text).strip())
        merged = " ".join(part for part in text_parts if part).strip()
        if merged:
            return merged
    return (raw_text or "").strip()


def _build_openai_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize parsed model segments into OpenAI-like segment objects."""
    openai_segments = []
    for idx, seg in enumerate(segments):
        start = _timestamp_to_seconds(seg.get("start_time"))
        end = _timestamp_to_seconds(seg.get("end_time"))
        item = {
            "id": idx,
            "start": start if start is not None else 0.0,
            "end": end if end is not None else (start if start is not None else 0.0),
            "text": str(seg.get("text", "") or ""),
        }
        speaker_id = seg.get("speaker_id")
        if speaker_id is not None:
            item["speaker"] = str(speaker_id)
        openai_segments.append(item)
    return openai_segments


def _fallback_segment(transcript_text: str, duration: Optional[float]) -> List[Dict[str, Any]]:
    end = duration if duration is not None and duration > 0 else 0.0
    return [{"id": 0, "start": 0.0, "end": end, "text": transcript_text}]


def _segments_to_srt(openai_segments: List[Dict[str, Any]], transcript_text: str, duration: Optional[float]) -> str:
    """Serialize transcript segments to SRT."""
    items = openai_segments if openai_segments else _fallback_segment(transcript_text, duration)
    lines: List[str] = []
    for idx, seg in enumerate(items, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        end = max(end, start)
        prefix = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        lines.append(str(idx))
        lines.append(f"{_format_timestamp_srt(start)} --> {_format_timestamp_srt(end)}")
        lines.append(f"{prefix}{seg.get('text', '')}".strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _segments_to_vtt(openai_segments: List[Dict[str, Any]], transcript_text: str, duration: Optional[float]) -> str:
    """Serialize transcript segments to WebVTT."""
    items = openai_segments if openai_segments else _fallback_segment(transcript_text, duration)
    lines: List[str] = ["WEBVTT", ""]
    for seg in items:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        end = max(end, start)
        prefix = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        lines.append(f"{_format_timestamp_vtt(start)} --> {_format_timestamp_vtt(end)}")
        lines.append(f"{prefix}{seg.get('text', '')}".strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _get_audio_duration_seconds(audio_path: str) -> Optional[float]:
    """Try reading the audio to compute duration for verbose responses."""
    try:
        audio, sample_rate = load_audio_use_ffmpeg(audio_path, resample=False)
        if sample_rate > 0:
            return float(len(audio)) / float(sample_rate)
    except Exception:
        return None
    return None


def create_openai_asr_app(
    asr: VibeVoiceASRBatchInference,
    *,
    served_model_name: str,
    default_max_new_tokens: int,
    default_top_p: float,
    default_num_beams: int,
) -> FastAPI:
    """
    Create an OpenAI-compatible ASR API app.

    Endpoint implemented:
      - POST /v1/audio/transcriptions
    """
    app = FastAPI(title="VibeVoice ASR OpenAI-Compatible API")
    inference_lock = threading.Lock()

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "owned_by": "vibevoice",
                }
            ],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(default=served_model_name),
        prompt: Optional[str] = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float = Form(default=0.0),
        language: Optional[str] = Form(default=None),
        max_new_tokens: int = Form(default=default_max_new_tokens),
        top_p: float = Form(default=default_top_p),
        num_beams: int = Form(default=default_num_beams),
    ):
        del language  # Model auto-detects language.
        requested_model = (model or served_model_name).strip() or served_model_name

        fmt = (response_format or "json").strip().lower()
        if fmt not in _SUPPORTED_AUDIO_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported response_format. Use one of: "
                    "json, text, srt, verbose_json, vtt, diarized_json."
                ),
            )
        allowed_formats = _allowed_formats_for_model(requested_model)
        if fmt not in allowed_formats:
            allowed_list = ", ".join(sorted(allowed_formats))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{requested_model}' supports response_format: {allowed_list}. "
                    f"Received: {fmt}"
                ),
            )
        if temperature < 0:
            raise HTTPException(status_code=400, detail="temperature must be >= 0")
        if max_new_tokens <= 0:
            raise HTTPException(status_code=400, detail="max_new_tokens must be > 0")
        if num_beams <= 0:
            raise HTTPException(status_code=400, detail="num_beams must be > 0")

        form_data = await request.form()
        timestamp_granularities = (
            form_data.getlist("timestamp_granularities[]")
            or form_data.getlist("timestamp_granularities")
        )

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = tmp.name

            with inference_lock:
                result = asr.transcribe_batch(
                    [temp_audio_path],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    num_beams=num_beams,
                    context_info=prompt,
                )[0]

            raw_text = result.get("raw_text", "")
            segments = result.get("segments", [])
            transcript_text = _segments_to_text(segments, raw_text)
            openai_segments = _build_openai_segments(segments)
            needs_duration = fmt in {"verbose_json", "srt", "vtt", "diarized_json"}
            duration = _get_audio_duration_seconds(temp_audio_path) if needs_duration else None

            if fmt == "text":
                return PlainTextResponse(transcript_text)

            if fmt == "srt":
                return PlainTextResponse(
                    _segments_to_srt(openai_segments, transcript_text, duration),
                    media_type="application/x-subrip",
                )

            if fmt == "vtt":
                return PlainTextResponse(
                    _segments_to_vtt(openai_segments, transcript_text, duration),
                    media_type="text/vtt",
                )

            if fmt == "diarized_json":
                diarized_segments = openai_segments if openai_segments else _fallback_segment(transcript_text, duration)
                normalized_diarized_segments = []
                for idx, seg in enumerate(diarized_segments):
                    item = dict(seg)
                    item["id"] = idx
                    item["speaker"] = str(item.get("speaker", "unknown"))
                    normalized_diarized_segments.append(item)
                diarized_payload: Dict[str, Any] = {
                    "text": transcript_text,
                    "segments": normalized_diarized_segments,
                }
                if duration is not None:
                    diarized_payload["duration"] = duration
                return JSONResponse(diarized_payload)

            if fmt == "verbose_json":
                payload: Dict[str, Any] = {
                    "task": "transcribe",
                    "text": transcript_text,
                    "segments": openai_segments,
                }
                if duration is not None:
                    payload["duration"] = duration

                if (
                    timestamp_granularities
                    and "word" in timestamp_granularities
                    and "words" not in payload
                ):
                    # Word-level timestamps are not natively produced by this model.
                    payload["words"] = []

                return JSONResponse(payload)

            return JSONResponse({"text": transcript_text})

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice ASR Batch Inference + OpenAI-Compatible API"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--audio_files", 
        type=str, 
        nargs='+',
        required=False,
        help="Paths to audio files for transcription"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=False,
        help="Directory containing audio files for batch transcription"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="HuggingFace dataset name (e.g., 'openslr/librispeech_asr')"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'test', 'test.other', 'test.clean')"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=3600.0,
        help="Maximum duration in seconds for concatenated dataset audio (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing multiple files"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else ("xpu" if torch.backends.xpu.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu") ),
        choices=["cuda", "cpu", "mps","xpu", "auto"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search. Use 1 for greedy/sampling"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Attention implementation to use. 'auto' will select the best available for your device (flash_attention_2 for CUDA, sdpa for MPS/CPU/XPU)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run an OpenAI-compatible ASR API server instead of batch file inference",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host for --serve mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for --serve mode",
    )
    parser.add_argument(
        "--served_model_name",
        type=str,
        default="vibevoice-asr",
        help="Model name returned by /v1/models in --serve mode",
    )
    
    args = parser.parse_args()

    if not args.model_path:
        print("Please provide --model_path, e.g. microsoft/VibeVoice-ASR")
        return
    
    # Auto-detect best attention implementation based on device
    if args.attn_implementation == "auto":
        if args.device == "cuda" and torch.cuda.is_available():
            try:
                import flash_attn
                args.attn_implementation = "flash_attention_2"
            except ImportError:
                print("flash_attn not installed, falling back to sdpa")
                args.attn_implementation = "sdpa"
        else:
            # MPS/XPU/CPU don't support flash_attention_2
            args.attn_implementation = "sdpa"
        print(f"Auto-detected attention implementation: {args.attn_implementation}")
    
    # Initialize model
    # Force float16 for model loading in this script.
    model_dtype = torch.float16
    
    asr = VibeVoiceASRBatchInference(
        model_path=args.model_path,
        device=args.device,
        dtype=model_dtype,
        attn_implementation=args.attn_implementation
    )

    if args.serve:
        import uvicorn

        app = create_openai_asr_app(
            asr=asr,
            served_model_name=args.served_model_name,
            default_max_new_tokens=args.max_new_tokens,
            default_top_p=args.top_p,
            default_num_beams=args.num_beams,
        )
        print(f"Starting OpenAI-compatible ASR server on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    # Collect audio files (batch mode)
    audio_files = []
    concatenated_audio = None  # For storing concatenated dataset audio

    if args.audio_files:
        audio_files.extend(args.audio_files)

    if args.audio_dir:
        import glob
        for ext in ["*.wav", "*.mp3", "*.flac", "*.mp4", "*.m4a", "*.webm"]:
            audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))

    if args.dataset:
        concatenated_audio = load_dataset_and_concatenate(
            dataset_name=args.dataset,
            split=args.split,
            max_duration=args.max_duration,
            num_audios=args.batch_size,
        )
        if concatenated_audio is None:
            return

    if len(audio_files) == 0 and concatenated_audio is None:
        print("No audio files provided. Please specify --audio_files, --audio_dir, or --dataset.")
        return

    if audio_files:
        print(f"\nAudio files to process ({len(audio_files)}):")
        for f in audio_files:
            print(f"  - {f}")

    if concatenated_audio:
        print(f"\nConcatenated dataset audios: {len(concatenated_audio)} audio(s)")
    
    # If temperature is 0, use greedy decoding (no sampling)
    do_sample = args.temperature > 0
    
    # Combine all audio inputs
    all_audio_inputs = audio_files + (concatenated_audio or [])
    
    print("\n" + "="*80)
    print(f"Processing {len(all_audio_inputs)} audio(s)")
    print("="*80)
    
    all_results = asr.transcribe_with_batching(
        all_audio_inputs,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=do_sample,
        num_beams=args.num_beams,
    )
    
    # Print results
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    for result in all_results:
        print("\n" + "-"*60)
        print_result(result)


if __name__ == "__main__":
    main()
