from typing import Dict, List, Optional

import pretty_midi as pm
import torch
from pydantic import BaseModel

from training.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class NoteData(BaseModel):
    onset_time: float
    offset_time: float
    pitch: int
    velocity: int


class PedalData(BaseModel):
    onset_time: float
    offset_time: float


def get_midi_events(midi_file: str):
    midi = pm.PrettyMIDI(midi_file)
    notes = [
        NoteData(
            onset_time=note.start,
            offset_time=note.end,
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in midi.instruments[0].notes
    ]
    pedals: List[PedalData] = []
    current_pedal: Optional[PedalData] = None

    for cc in sorted(midi.instruments[0].control_changes, key=lambda x: x.time):
        if cc.number != 64:
            continue
        if cc.value > 64:
            if current_pedal is not None:
                continue
            current_pedal = PedalData(onset_time=cc.time, offset_time=0)
        else:
            if current_pedal is None and len(pedals) > 0:
                pedals[-1].offset_time = cc.time
                continue
            elif current_pedal is None:
                continue
            current_pedal.offset_time = cc.time
            pedals.append(
                PedalData(
                    onset_time=current_pedal.onset_time,
                    offset_time=current_pedal.offset_time,
                )
            )
            current_pedal = None

    return notes, pedals


def get_regression(x: torch.Tensor, config: Config):
    step = 1.0 / config.frames_per_second
    output = torch.zeros_like(x)

    locts = torch.where(x < 0.5)[0]

    if len(locts) > 0:
        for t in range(0, locts[0]):
            output[t] = step * (t - locts[0]) - x[locts[0]]

        for i in range(0, len(locts) - 1):
            for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                output[t] = step * (t - locts[i]) - x[locts[i]]

            for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                output[t] = step * (t - locts[i + 1]) - x[locts[i + 1]]

        for t in range(locts[-1], len(x)):
            output[t] = step * (t - locts[-1]) - x[locts[-1]]

    output = torch.clip(torch.abs(output), 0.0, 0.05) * 20
    output = 1.0 - output

    return output


def process_midi_events(
    notes: List[NoteData],
    pedals: List[PedalData],
    config: Config,
    start_time: float,
    end_time: float,
    pitch_shift: int = 0,
):
    segmented_notes: List[NoteData] = []
    segmenteed_pedals: List[PedalData] = []

    unpaired_notes: Dict[int, NoteData] = {}

    for note in notes:
        if (
            start_time <= note.onset_time < end_time
            and start_time < note.offset_time < end_time
        ):
            segmented_notes.append(note)
        elif start_time <= note.onset_time < end_time and note.offset_time > end_time:
            note.offset_time = end_time
            segmented_notes.append(note)
            unpaired_notes[note.pitch] = note

    for pedal in pedals:
        if (
            start_time <= pedal.onset_time < end_time
            and start_time < pedal.offset_time <= end_time
        ):
            pedal.offset_time = end_time
            segmenteed_pedals.append(pedal)

    num_frames = int(round(config.segment_seconds * config.frames_per_second)) + 1
    onset_roll = torch.zeros((num_frames, config.midi.num_notes))
    offset_roll = torch.zeros((num_frames, config.midi.num_notes))
    reg_onset_roll = torch.ones((num_frames, config.midi.num_notes))
    reg_offset_roll = torch.ones((num_frames, config.midi.num_notes))
    frame_roll = torch.zeros((num_frames, config.midi.num_notes))
    velocity_roll = torch.zeros((num_frames, config.midi.num_notes))
    mask_roll = torch.ones((num_frames, config.midi.num_notes))

    pedal_onset_roll = torch.zeros(num_frames)
    pedal_offset_roll = torch.zeros(num_frames)
    reg_pedal_onset_roll = torch.ones(num_frames)
    reg_pedal_offset_roll = torch.ones(num_frames)
    pedal_frame_roll = torch.zeros(num_frames)

    for note in segmented_notes:
        pitch = torch.clip(
            torch.tensor(note.pitch - config.midi.begin_note + pitch_shift),
            0,
            config.midi.num_notes - 1,
        )

        if 0 <= pitch < config.midi.num_notes:
            begin_frame = int(
                round((note.onset_time - start_time) * config.frames_per_second)
            )
            end_frame = int(
                round((note.offset_time - start_time) * config.frames_per_second)
            )

            if end_frame >= 0:
                frame_roll[max(begin_frame, 0) : end_frame + 1, pitch] = 1
                offset_roll[end_frame, pitch] = 1
                velocity_roll[max(begin_frame, 0) : end_frame + 1, pitch] = (
                    note.velocity
                )

                reg_offset_roll[end_frame, pitch] = (note.offset_time - start_time) - (
                    end_frame / config.frames_per_second
                )

                if begin_frame >= 0:
                    onset_roll[begin_frame, pitch] = 1
                    reg_onset_roll[begin_frame, pitch] = (
                        note.onset_time - start_time
                    ) - (begin_frame / config.frames_per_second)
                else:
                    mask_roll[: end_frame + 1, pitch] = 0

    for k in range(config.midi.num_notes):
        reg_onset_roll[:, k] = get_regression(reg_onset_roll[:, k], config)
        reg_offset_roll[:, k] = get_regression(reg_offset_roll[:, k], config)

    for pitch, note in unpaired_notes.items():
        pitch = torch.clip(
            torch.tensor(note.pitch - config.midi.begin_note + pitch_shift),
            0,
            config.midi.num_notes - 1,
        )

        if 0 <= pitch < config.midi.num_notes:
            begin_frame = int(
                round((note.onset_time - start_time) * config.frames_per_second)
            )
            mask_roll[begin_frame:, pitch] = 0

    for pedal in segmenteed_pedals:
        begin_frame = int(
            round((pedal.onset_time - start_time) * config.frames_per_second)
        )
        end_frame = int(
            round((pedal.offset_time - start_time) * config.frames_per_second)
        )

        if end_frame >= 0:
            pedal_frame_roll[max(begin_frame, 0) : end_frame + 1] = 1

            pedal_offset_roll[end_frame] = 1
            reg_pedal_offset_roll[end_frame] = (pedal.offset_time - start_time) - (
                end_frame / config.frames_per_second
            )

            if begin_frame >= 0:
                pedal_onset_roll[begin_frame] = 1
                reg_pedal_onset_roll[begin_frame] = (pedal.onset_time - start_time) - (
                    begin_frame / config.frames_per_second
                )

    reg_pedal_onset_roll = get_regression(reg_pedal_onset_roll, config)
    reg_pedal_offset_roll = get_regression(reg_pedal_offset_roll, config)

    return {
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "reg_onset_roll": reg_onset_roll,
        "reg_offset_roll": reg_offset_roll,
        "frame_roll": frame_roll,
        "velocity_roll": velocity_roll,
        "mask_roll": mask_roll,
        "reg_pedal_onset_roll": reg_pedal_onset_roll,
        "pedal_onset_roll": pedal_onset_roll,
        "pedal_offset_roll": pedal_offset_roll,
        "reg_pedal_offset_roll": reg_pedal_offset_roll,
        "pedal_frame_roll": pedal_frame_roll,
    }
