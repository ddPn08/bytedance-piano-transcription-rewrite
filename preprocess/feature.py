from typing import Optional

import torch
import torchaudio

from training.config import Config


def load_audio(
    wav_path: str,
    sampling_rate: int,
):
    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(dim=0)
    if sr != sampling_rate:
        audio = torchaudio.functional.resample(
            audio, sr, sampling_rate, resampling_method="sinc_interp_kaiser"
        )

    return audio


def create_mel_transform(
    config: Config,
    device=torch.device("cpu"),
):
    hop_size = config.feature.sampling_rate // config.frames_per_second
    f_max = config.feature.sampling_rate // 2

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=config.feature.sampling_rate,
        n_fft=config.feature.win_length,
        win_length=config.feature.win_length,
        hop_length=hop_size,
        center=config.feature.center,
        pad_mode=config.feature.pad_mode,
        n_mels=config.feature.num_mels,
        f_min=config.feature.f_min,
        f_max=f_max,
        norm="slaney",
        mel_scale="slaney",
    ).to(device)


def create_feature(
    audio: torch.Tensor,
    config: Config,
    device=torch.device("cpu"),
    mel_transform: Optional[torchaudio.transforms.MelSpectrogram] = None,
):
    if mel_transform is None:
        mel_transform = create_mel_transform(config, device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.float()
    audio = audio.to(device)

    x = mel_transform(audio)

    x = x.permute(0, 2, 1)
    x = 10.0 * torch.log10(torch.clamp(x, min=config.feature.amin, max=torch.inf))
    x -= 10.0 * torch.log10(
        torch.max(torch.tensor(config.feature.amin), torch.tensor(config.feature.ref))
    )

    return x
