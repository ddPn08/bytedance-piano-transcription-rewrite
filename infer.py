import json
from typing import Optional

import fire
import numpy as np
import torch
import tqdm

from modules.note import RegressNoteModel
from preprocess.feature import create_feature, create_mel_transform, load_audio
from preprocess.midi import RegressionPostProcessor, events_to_midi
from training.config import Config


def fix_model_state_dict(state_dict):
    if "model" in state_dict:
        state_dict = state_dict["model"]["note_model"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("spectrogram_extractor."):
                continue
            elif k.startswith("logmel_extractor."):
                continue
            if k.startswith("frame_model."):
                k = k.replace("frame_model.", "frame.")
            elif k.startswith("reg_onset_model."):
                k = k.replace("reg_onset_model.", "onset.")
            elif k.startswith("reg_offset_model."):
                k = k.replace("reg_offset_model.", "offset.")
            elif k.startswith("velocity_model."):
                k = k.replace("velocity_model.", "velocity.")
            elif k.startswith("reg_onset_gru."):
                k = k.replace("reg_onset_gru.", "onset_gru.")
            elif k.startswith("reg_onset_fc"):
                k = k.replace("reg_onset_fc", "onset_fc")
            new_state_dict[k] = v
        state_dict = new_state_dict
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key.replace("model.", ""): value for key, value in state_dict.items()
        }
    return state_dict


def main(
    wav_path: str,
    output_path: str,
    note_model_path: str,
    config_path: str = "config.json",
    pedal_motel_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 1,
):
    with open(config_path, "r") as f:
        config = Config.model_validate(json.load(f))
    device = torch.device(device)
    segment_samples = int(config.feature.sampling_rate * config.segment_seconds)

    note_model_state_dict = fix_model_state_dict(
        torch.load(note_model_path, weights_only=True)
    )
    note_model = RegressNoteModel(config.midi.num_notes, config.feature.num_mels).to(
        device
    )
    note_model.load_state_dict(note_model_state_dict)
    note_model.eval()

    audio = load_audio(
        wav_path,
        sampling_rate=config.feature.sampling_rate,
    )
    # audio = audio[100000:130000]
    audio = audio[None, :]
    audio_len = audio.shape[1]
    pad_len = int(np.ceil(audio_len / segment_samples)) * segment_samples - audio_len

    # audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)
    audio = torch.cat((audio, torch.zeros((1, pad_len))), dim=1)

    assert audio.shape[1] % segment_samples == 0
    batch = []

    pointer = 0
    while pointer + segment_samples <= audio.shape[1]:
        batch.append(audio[:, pointer : pointer + segment_samples])
        pointer += segment_samples // 2

    batch = torch.cat(batch, dim=0)

    mel_transform = create_mel_transform(config, device)

    output_dict = {}

    for i in tqdm.tqdm(range(0, len(batch), batch_size)):
        with torch.no_grad():
            feature = create_feature(
                batch[i : i + batch_size], config, device, mel_transform
            )
            feature = feature.unsqueeze(1)
            output = note_model(feature)

            for k, v in output.items():
                if k not in output_dict:
                    output_dict[k] = []
                output_dict[k].append(v.cpu().numpy())

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    for k, v in output_dict.items():
        if v.shape[0] == 1:
            output_dict[k] = v[0]
        else:
            v = v[:, 0:-1, :]
            """Remove an extra frame in the end of each segment caused by the
            'center=True' argument when calculating spectrogram."""
            (N, segment_samples, classes_num) = v.shape
            assert segment_samples % 4 == 0

            y = []
            y.append(v[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(
                    v[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)]
                )
            y.append(v[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            output_dict[k] = y

    onset_threshold = 0.3
    offset_threshod = 0.3
    frame_threshold = 0.1
    pedal_offset_threshold = 0.2

    post_processor = RegressionPostProcessor(
        config,
        onset_threshold,
        offset_threshod,
        frame_threshold,
        pedal_offset_threshold,
    )

    (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
        output_dict
    )

    midi = events_to_midi(est_note_events, est_pedal_events)
    midi.write(output_path)


if __name__ == "__main__":
    fire.Fire(main)
