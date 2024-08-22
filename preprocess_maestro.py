import json
import multiprocessing as mp
import os
from typing import List

import fire
import torch
import tqdm
from pydantic import  RootModel

from preprocess.feature import load_audio
from preprocess.midi import get_midi_events, process_midi_events
from training.config import Config
from training.dataset import Metadata, Segment
from utils.logger import get_logger

logger = get_logger(__name__)



def process_data(
    idx: int,
    segments: List[Segment],
    dataset_path: str,
    config: Config,
    audio_dir: str,
    midi_dir: str,
    force_preprocess: bool,
):
    for segment in tqdm.tqdm(segments, desc=f"ProcessData {idx}", position=idx):
        audio_basename = os.path.basename(segment.audio_filename.replace("/", "-"))
        midi_basename = os.path.basename(segment.midi_filename.replace("/", "-"))

        wav_data_path = os.path.join(
            audio_dir, segment.split, f"{segment.start_time}-{audio_basename}.pt"
        )
        midi_data_path = os.path.join(
            midi_dir, segment.split, f"{segment.start_time}-{midi_basename}.pt"
        )
        os.makedirs(os.path.dirname(wav_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(midi_data_path), exist_ok=True)
        try:
            if not os.path.exists(wav_data_path) or force_preprocess:
                audio = load_audio(
                    os.path.join(dataset_path, segment.audio_filename),
                    sampling_rate=config.feature.sampling_rate,
                )
                start_frame = int(segment.start_time * config.feature.sampling_rate)
                end_frame = start_frame + int(
                    config.segment_seconds * config.feature.sampling_rate
                )
                torch.save(audio[start_frame:end_frame], wav_data_path)

            if not os.path.exists(midi_data_path) or force_preprocess:
                notes, pedals = get_midi_events(
                    os.path.join(dataset_path, segment.midi_filename),
                )
                data = process_midi_events(
                    notes,
                    pedals,
                    config,
                    segment.start_time,
                    segment.start_time + config.segment_seconds,
                )
                torch.save(data, midi_data_path)

        except Exception as e:
            logger.error(f"Error: {midi_basename}")
            logger.error(e)
            raise e


def main(
    config_path: str = "config.json",
    dataset_path: str = "maestro-v3.0.0",
    dest_path: str = "maestro-v3.0.0-preprocessed",
    num_workers: int = 4,
    force_preprocess: bool = False,
):
    with open(config_path, "r") as f:
        config = Config.model_validate(json.load(f))

    with open(os.path.join(dataset_path, "maestro-v3.0.0.json"), "r") as f:
        raw_metadata = json.load(f)

    metadata: List[Metadata] = []
    keys = list(raw_metadata.keys())

    for idx in range(len(raw_metadata[keys[0]])):
        data = {}
        for key in keys:
            data[key] = raw_metadata[key][str(idx)]
        metadata.append(Metadata.model_validate(data))

    segments: List[Segment] = []

    for idx, m in enumerate(metadata):
        start_time = 0.0

        while start_time + config.segment_seconds < m.duration:
            segment = Segment(
                year=m.year,
                midi_filename=m.midi_filename,
                audio_filename=m.audio_filename,
                split=m.split,
                start_time=start_time,
            )
            segments.append(segment)
            start_time += config.segment_seconds

    midi_dir = os.path.join(dest_path, "midi")
    audio_dir = os.path.join(dest_path, "audio")
    os.makedirs(midi_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    processes = []

    for idx in range(num_workers):
        p = mp.Process(
            target=process_data,
            args=(
                idx,
                segments[idx::num_workers],
                dataset_path,
                config,
                audio_dir,
                midi_dir,
                force_preprocess,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    config_path = os.path.join(dest_path, "config.json")
    with open(config_path, "w") as f:
        f.write(config.model_dump_json(indent=4))

    metadata_path = os.path.join(dest_path, "metadata.json")
    with open(metadata_path, "w") as f:
        f.write(RootModel(metadata).model_dump_json())

    segments_path = os.path.join(dest_path, "segments.json")
    with open(segments_path, "w") as f:
        f.write(RootModel(segments).model_dump_json())


if __name__ == "__main__":
    fire.Fire(main)
