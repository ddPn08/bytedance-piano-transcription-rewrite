import json
import os
import pickle
from typing import Dict, List, Literal

import torch
import torch.utils.data as data
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
    year: int
    midi_filename: str
    audio_filename: str
    split: str
    start_time: float


class Dataset(data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: Literal["train", "validation", "test"] = "train",
        generator: torch.Generator = torch.Generator(),
    ):
        self.dataset_path = dataset_path
        config_path = os.path.join(dataset_path, "config.json")
        metadata_path = os.path.join(dataset_path, "metadata.json")
        segments_path = os.path.join(dataset_path, "segments.json")
        with open(config_path, "r") as f:
            self.config = Config.model_validate(json.load(f))
        with open(metadata_path, "r") as f:
            self.metadata = [Metadata.model_validate(m) for m in json.load(f)]
        with open(segments_path, "r") as f:
            self.segments = [Segment.model_validate(s) for s in json.load(f)]

        self.metadata = [m for m in self.metadata if m.split == split]

        self.generator = generator

    def __getitem__(self, idx: int):
        segment = self.segments[idx]
        
        audio_basename = os.path.basename(segment.audio_filename.replace("/", "-"))
        midi_basename = os.path.basename(segment.midi_filename.replace("/", "-"))

        audio_path = os.path.join(
            self.dataset_path,
            "audio",
            segment.split,
            f"{segment.start_time}-{audio_basename}.pt",
        )
        audio = torch.load(audio_path, weights_only=True)

        midi_path = os.path.join(
            self.dataset_path,
            "midi",
            segment.split,
            f"{segment.start_time}-{midi_basename}.pt",
        )
        midi_data = torch.load(midi_path, weights_only=True)

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
