import json
import multiprocessing as mp
import os
import pickle
from typing import List

import fire
import torch
import tqdm
from pydantic import RootModel

from preprocess.feature import load_audio
from preprocess.midi import get_midi_events
from training.config import Config
from training.dataset import Metadata
from utils.logger import get_logger

logger = get_logger(__name__)


def process_data(
    idx: int,
    metadata: List[Metadata],
    dataset_path: str,
    config: Config,
    audio_dir: str,
    midi_dir: str,
    force_preprocess: bool,
):
    for m in tqdm.tqdm(metadata, desc=f"ProcessData {idx}", position=idx):
        audio_basename = os.path.basename(m.audio_filename.replace("/", "-"))
        midi_basename = os.path.basename(m.midi_filename.replace("/", "-"))

        wav_data_path = os.path.join(audio_dir, m.split, f"{audio_basename}.pt")
        midi_data_path = os.path.join(midi_dir, m.split, f"{midi_basename}.pkl")
        os.makedirs(os.path.dirname(wav_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(midi_data_path), exist_ok=True)
        try:
            if not os.path.exists(wav_data_path) or force_preprocess:
                audio = load_audio(
                    os.path.join(dataset_path, m.audio_filename),
                    sampling_rate=config.feature.sampling_rate,
                )
                torch.save(audio, wav_data_path)

            if not os.path.exists(midi_data_path) or force_preprocess:
                notes, pedals = get_midi_events(
                    os.path.join(dataset_path, m.midi_filename),
                )
                with open(midi_data_path, "wb") as f:
                    pickle.dump(
                        {
                            "notes": notes,
                            "pedals": pedals,
                        },
                        f,
                    )

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
                metadata[idx::num_workers],
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


if __name__ == "__main__":
    fire.Fire(main)
