import librosa
import numpy as np
import torch
import torchaudio
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

frames_per_second = 100
sample_rate = 16000
window_size = 2048
hop_size = sample_rate // frames_per_second
mel_bins = 229
fmin = 30
fmax = sample_rate // 2

window = "hann"
center = True
pad_mode = "reflect"
ref = 1.0
amin = 1e-10
top_db = None


def calc_log_melspec_with_torchlibrosa(wav_path: str):
    # audio, _ = librosa.core.load(wav_path, sr=sample_rate, mono=True)
    # audio = torch.from_numpy(audio)

    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(dim=0)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    audio = audio.unsqueeze(0)
    audio = audio.float()

    spectrogram_extractor = Spectrogram(
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        center=center,
        pad_mode=pad_mode,
        freeze_parameters=True,
    )

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(
        sr=sample_rate,
        n_fft=window_size,
        n_mels=mel_bins,
        fmin=fmin,
        fmax=fmax,
        ref=ref,
        amin=amin,
        top_db=top_db,
        freeze_parameters=True,
        is_log=True,
    )
    x = spectrogram_extractor(audio)
    x = logmel_extractor(x)
    return x


def calc_log_melspec_with_torchaudio(wav_path: str):
    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(dim=0)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    audio = audio.unsqueeze(0)
    audio = audio.float()
    x = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=window_size,
        win_length=window_size,
        hop_length=hop_size,
        center=center,
        pad_mode=pad_mode,
        n_mels=mel_bins,
        f_min=fmin,
        f_max=fmax,
        norm="slaney",
        mel_scale="slaney",
    )(audio)
    x = x.permute(0, 2, 1)
    x = 10.0 * torch.log10(torch.clamp(x, min=amin, max=torch.inf))
    x -= 10.0 * torch.log10(torch.max(torch.tensor(amin), torch.tensor(ref)))
    return x


def main():
    # wav_path = "/mnt/s/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
    wav_path = "/mnt/s/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_02_Track02_wav.wav"
    x1 = calc_log_melspec_with_torchlibrosa(wav_path)
    x2 = calc_log_melspec_with_torchaudio(wav_path)

    print(x1.shape)  # torch.Size([1, 1, *, *])
    print(x2.shape)  # torch.Size([1, *, *])
    # torch.Size([1, *, *]) -> torch.Size([1, 1, *, *])
    x2 = x2.unsqueeze(1)

    # assert x1 = x2
    print(torch.allclose(x1, x2, atol=1e-5))

    print(x1[0, 0, :10, :10])
    print(x2[0, 0, :10, :10])


if __name__ == "__main__":
    with torch.no_grad():
        main()
