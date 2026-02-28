from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def _resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    if explicit_token:
        return explicit_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _move_to_device(value: Any, device: torch.device):
    if hasattr(value, "to"):
        return value.to(device)
    return value


@dataclass(frozen=True)
class AudioResult:
    audio: np.ndarray
    sample_rate: int = 24_000
    text: Optional[str] = None


class ChromaInference:
    def __init__(
        self,
        model_id: str = "FlashLabs/Chroma-4B",
        *,
        device_map: str | dict | None = "auto",
        trust_remote_code: bool = True,
        token: Optional[str] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        hf_token = _resolve_hf_token(token)
        model_kwargs = dict(model_kwargs or {})
        processor_kwargs = dict(processor_kwargs or {})

        # `token` is supported by recent `transformers`/`huggingface_hub`.
        # If not needed, passing None is fine.
        if hf_token is not None:
            model_kwargs.setdefault("token", hf_token)
            processor_kwargs.setdefault("token", hf_token)

        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        processor_kwargs.setdefault("trust_remote_code", trust_remote_code)

        if device_map is not None:
            model_kwargs.setdefault("device_map", device_map)

        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    def load_prompt(
        self,
        speaker_name: str,
        *,
        base_dir: Union[str, Path] = "example",
    ) -> Tuple[list[str], list[str]]:
        base_dir = Path(base_dir)
        text_path = base_dir / "prompt_text" / f"{speaker_name}.txt"
        audio_path = base_dir / "prompt_audio" / f"{speaker_name}.wav"

        prompt_text = text_path.read_text(encoding="utf-8")
        return [prompt_text], [str(audio_path)]

    def generate_audio(
        self,
        *,
        input_audio: Union[str, np.ndarray],
        speaker_name: str,
        system_prompt: Optional[str] = None,
        prompt_text: Optional[list[str]] = None,
        prompt_audio: Optional[list[str]] = None,
        return_text: bool = False,
        max_new_text_tokens: int = 128,
        max_new_tokens: int = 1000,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> AudioResult:
        if system_prompt is None:
            system_prompt = (
                "You are Chroma, an advanced virtual human created by the FlashLabs. "
                "You possess the ability to understand auditory inputs and generate both text and speech."
            )

        if prompt_text is None or prompt_audio is None:
            prompt_text, prompt_audio = self.load_prompt(speaker_name)

        conversation = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": input_audio}],
                },
            ]
        ]

        inputs = self.processor(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
        )

        device = self.model.device
        inputs = {k: _move_to_device(v, device) for k, v in inputs.items()}

        generated_text: Optional[str] = None
        if return_text:
            generated_text = self._generate_text_from_thinker(inputs, max_new_text_tokens=max_new_text_tokens)

        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
        )

        decoded = self.model.codec_model.decode(output_tokens.permute(0, 2, 1))
        audio_values = decoded.audio_values[0].cpu().detach().numpy()
        return AudioResult(audio=audio_values, sample_rate=24_000, text=generated_text)

    @torch.inference_mode()
    def _generate_text_from_thinker(self, model_inputs: dict[str, Any], *, max_new_text_tokens: int) -> Optional[str]:
        """Generate a text response using the internal 'thinker' model.

        Notes:
        - Chroma's audio generation loop uses thinker tokens internally but does not expose them.
          This helper runs the thinker generation separately to obtain a text response.
        - The text returned may not be bit-identical to any implicit text used during audio generation.
        """

        thinker_input_ids = model_inputs.get("thinker_input_ids")
        if thinker_input_ids is None:
            return None

        thinker_attention_mask = model_inputs.get("thinker_attention_mask")
        thinker_input_features = model_inputs.get("thinker_input_features")
        thinker_feature_attention_mask = model_inputs.get("thinker_feature_attention_mask")

        thinker_generate_kwargs: dict[str, Any] = {
            "input_ids": thinker_input_ids,
            "attention_mask": thinker_attention_mask,
            "input_features": thinker_input_features,
            "feature_attention_mask": thinker_feature_attention_mask,
            "use_cache": True,
            "max_new_tokens": int(max_new_text_tokens),
            # The model's internal audio loop uses argmax; greedy generation matches that behavior best.
            "do_sample": False,
            # Some Qwen2.5-Omni variants accept this flag.
            "use_audio_in_video": False,
        }

        # Prefer the model's configured end token if available.
        eos_token_id = getattr(self.model.config, "im_end_token_id", None)
        if eos_token_id is not None:
            thinker_generate_kwargs.setdefault("eos_token_id", eos_token_id)

        thinker_out = self.model.thinker.generate(**thinker_generate_kwargs)

        # Decode only newly generated tokens (exclude the prompt).
        prompt_len = thinker_input_ids.shape[1]
        new_token_ids = thinker_out[:, prompt_len:]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None

        text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]
        return text.strip() if isinstance(text, str) else None
