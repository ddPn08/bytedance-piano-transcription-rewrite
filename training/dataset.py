import json
import os
import pickle
from typing import Dict, List, Literal

import torch
import torch.utils.data as data
import torchaudio
from pydantic import BaseModel

from preprocess.midi import NoteData, PedalData, process_midi_events
from training.config import Config


class Metadata(BaseModel):
    canonical_composer: str
    canonical_title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float


class Segment(BaseModel):
    idx: int
    start: float


class Dataset(data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: Literal["train", "validation", "test"] = "train",
        generator: torch.Generator = torch.Generator(),
        max_pitch_shift: int = 0,
    ):
        self.dataset_path = dataset_path
        config_path = os.path.join(dataset_path, "config.json")
        metadata_path = os.path.join(dataset_path, "metadata.json")
        with open(config_path, "r") as f:
            self.config = Config.model_validate(json.load(f))
        with open(metadata_path, "r") as f:
            self.metadata = [Metadata.model_validate(m) for m in json.load(f)]

        self.metadata = [m for m in self.metadata if m.split == split]

        self.generator = generator

        self.max_pitch_shift = max_pitch_shift
        self.segment_samples = int(
            self.config.feature.sampling_rate * self.config.segment_seconds
        )

        self.segments: List[Segment] = []

        for idx, m in enumerate(self.metadata):
            start_time = 0.0

            while start_time + self.config.segment_seconds < m.duration:
                segment = Segment(idx=idx, start=start_time)
                self.segments.append(segment)
                start_time += self.config.hop_seconds

    def __getitem__(self, idx: int):
        segment = self.segments[idx]
        metadata = self.metadata[segment.idx]
        audio_path = os.path.join(
            self.dataset_path,
            "audio",
            metadata.split,
            metadata.audio_filename.replace("/", "-") + ".pt",
        )
        audio = torch.load(audio_path, weights_only=True)

        midi_path = os.path.join(
            self.dataset_path,
            "midi",
            metadata.split,
            metadata.midi_filename.replace("/", "-") + ".pkl",
        )
        with open(midi_path, "rb") as f:
            midi = pickle.load(f)
            notes: List[NoteData] = midi["notes"]
            pedals: List[PedalData] = midi["pedals"]

        pitch_shift = torch.randint(
            -self.max_pitch_shift,
            self.max_pitch_shift + 1,
            (1,),
            generator=self.generator,
        ).item()

        start_sample = int(segment.start * self.config.feature.sampling_rate)
        end_sample = start_sample + self.segment_samples

        if end_sample >= audio.size(0):
            start_sample -= self.segment_samples
            end_sample -= self.segment_samples

        audio = audio[start_sample:end_sample]

        if pitch_shift != 0:
            audio = torchaudio.functional.pitch_shift(
                audio, self.config.feature.sampling_rate, pitch_shift
            )

        midi_data = process_midi_events(
            notes,
            pedals,
            self.config,
            segment.start,
            segment.start + self.config.segment_seconds,
            pitch_shift=pitch_shift,
        )

        return {
            "audio": audio,
            **midi_data,
        }

    def __len__(self):
        return len(self.segments)

    def collate_fn(self, batch: List[Dict]):
        return (
            torch.stack([data["audio"] for data in batch]),
            torch.stack([data["onset_roll"] for data in batch]),
            torch.stack([data["offset_roll"] for data in batch]),
            torch.stack([data["reg_onset_roll"] for data in batch]),
            torch.stack([data["reg_offset_roll"] for data in batch]),
            torch.stack([data["frame_roll"] for data in batch]),
            torch.stack([data["velocity_roll"] for data in batch]),
            torch.stack([data["mask_roll"] for data in batch]),
            torch.stack([data["pedal_onset_roll"] for data in batch]),
            torch.stack([data["pedal_offset_roll"] for data in batch]),
            torch.stack([data["reg_pedal_onset_roll"] for data in batch]),
            torch.stack([data["reg_pedal_offset_roll"] for data in batch]),
            torch.stack([data["pedal_frame_roll"] for data in batch]),
        )
