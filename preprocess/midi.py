from typing import Dict, List, Optional

import numpy as np
import pretty_midi as pm
import torch
from pydantic import BaseModel
from rust_ext import calc_regression

from training.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


def get_regression(x: torch.Tensor, frames_per_second: int):
    x = x.cpu().numpy()
    return torch.tensor(calc_regression(x, frames_per_second))


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
        reg_onset_roll[:, k] = get_regression(
            reg_onset_roll[:, k], config.frames_per_second
        )
        reg_offset_roll[:, k] = get_regression(
            reg_offset_roll[:, k], config.frames_per_second
        )

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

    reg_pedal_onset_roll = get_regression(
        reg_pedal_onset_roll, config.frames_per_second
    )
    reg_pedal_offset_roll = get_regression(
        reg_pedal_offset_roll, config.frames_per_second
    )

    return {
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "reg_onset_roll": reg_onset_roll,
        "reg_offset_roll": reg_offset_roll,
        "frame_roll": frame_roll,
        "velocity_roll": velocity_roll,
        "mask_roll": mask_roll,
        "pedal_onset_roll": pedal_onset_roll,
        "pedal_offset_roll": pedal_offset_roll,
        "reg_pedal_onset_roll": reg_pedal_onset_roll,
        "reg_pedal_offset_roll": reg_pedal_offset_roll,
        "pedal_frame_roll": pedal_frame_roll,
    }


def note_detection_with_onset_offset_regress(
    frame_output: torch.Tensor,
    onset_output: torch.Tensor,
    onset_shift_output: torch.Tensor,
    offset_output: torch.Tensor,
    offset_shift_output: torch.Tensor,
    velocity_output: torch.Tensor,
    frame_threshold: float,
):
    """Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity],
      e.g., [
        [1821, 1909, 0.47498, 0.3048533, 0.72119445],
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            """Onset detected"""
            if bgn:
                """Consecutive onsets. E.g., pedal is not released, but two 
                consecutive notes being played."""
                fin = max(i - 1, 0)
                output_tuples.append(
                    [bgn, fin, onset_shift_output[bgn], 0, velocity_output[bgn]]
                )
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    """bgn --------- offset_occur --- frame_disappear"""
                    fin = offset_occur
                else:
                    """bgn --- offset_occur --------- frame_disappear"""
                    fin = frame_disappear
                output_tuples.append(
                    [
                        bgn,
                        fin,
                        onset_shift_output[bgn],
                        offset_shift_output[fin],
                        velocity_output[bgn],
                    ]
                )
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                """Offset not detected"""
                fin = i
                output_tuples.append(
                    [
                        bgn,
                        fin,
                        onset_shift_output[bgn],
                        offset_shift_output[fin],
                        velocity_output[bgn],
                    ]
                )
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def pedal_detection_with_onset_offset_regress(
    frame_output: torch.Tensor,
    offset_output: torch.Tensor,
    offset_shift_output: torch.Tensor,
    frame_threshold: float,
):
    """Process prediction array to pedal events information.

    Args:
      frame_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      frame_threshold: float

    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift],
      e.g., [
        [1821, 1909, 0.4749851, 0.3048533],
        [1909, 1947, 0.30730522, -0.45764327],
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            """Pedal onset detected"""
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                """offset not detected but frame disappear"""
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


class RegressionPostProcessor:
    def __init__(
        self,
        config: Config,
        onset_threshold: float,
        offset_threshold: float,
        frame_threshold: float,
        pedal_offset_threshold: float,
    ):
        """Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
          pedal_offset_threshold: float
        """
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.midi.num_notes
        self.begin_note = config.midi.begin_note
        self.velocity_scale = config.midi.velocity_scale

        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num),
            'reg_offset_output': (segment_frames, classes_num),
            'frame_output': (segment_frames, classes_num),
            'velocity_output': (segment_frames, classes_num),
            'reg_pedal_onset_output': (segment_frames, 1),
            'reg_pedal_offset_output': (segment_frames, 1),
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83},
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96},
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = (
            self.output_dict_to_note_pedal_arrays(output_dict)
        )
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity], 
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""

        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        """Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num),
            'reg_offset_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'velocity_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time,
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65],
             [11.98, 12.11, 33, 0.69],
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time
            and offset_time. E.g. [
             [0.17, 0.96],
             [1.17, 2.65],
             ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = self.get_binarized_output_from_regression(
            reg_output=output_dict["reg_onset_output"],
            threshold=self.onset_threshold,
            neighbour=2,
        )

        output_dict["onset_output"] = onset_output  # Values are 0 or 1
        output_dict["onset_shift_output"] = onset_shift_output

        # Calculate binarized offset output from regression output
        (offset_output, offset_shift_output) = (
            self.get_binarized_output_from_regression(
                reg_output=output_dict["reg_offset_output"],
                threshold=self.offset_threshold,
                neighbour=4,
            )
        )

        output_dict["offset_output"] = offset_output  # Values are 0 or 1
        output_dict["offset_shift_output"] = offset_shift_output

        if "reg_pedal_onset_output" in output_dict.keys():
            """Pedal onsets are not used in inference. Instead, frame-wise pedal
            predictions are used to detect onsets. We empirically found this is 
            more accurate to detect pedal onsets."""
            pass

        if "reg_pedal_offset_output" in output_dict.keys():
            # Calculate binarized pedal offset output from regression output
            (pedal_offset_output, pedal_offset_shift_output) = (
                self.get_binarized_output_from_regression(
                    reg_output=output_dict["reg_pedal_offset_output"],
                    threshold=self.pedal_offset_threshold,
                    neighbour=4,
                )
            )

            output_dict["pedal_offset_output"] = (
                pedal_offset_output  # Values are 0 or 1
            )
            output_dict["pedal_offset_shift_output"] = pedal_offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)

        if "reg_pedal_onset_output" in output_dict.keys():
            # Detect piano pedals from output_dict
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)

        else:
            est_pedal_on_offs = None

        return est_on_off_note_vels, est_pedal_on_offs

    def get_binarized_output_from_regression(
        self, reg_output: torch.Tensor, threshold: float, neighbour: int
    ):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int

        Returns:
          binary_output: (frames_num, classes_num)
          shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        # binary_output = torch.zeros_like(reg_output)
        # shift_output = torch.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape

        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription 
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x: torch.Tensor, n: torch.Tensor, neighbour: int):
        """Detect if values are monotonic in both side of x[n].

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets,
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict["frame_output"].shape[-1]

        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = note_detection_with_onset_offset_regress(
                frame_output=output_dict["frame_output"][:, piano_note],
                onset_output=output_dict["onset_output"][:, piano_note],
                onset_shift_output=output_dict["onset_shift_output"][:, piano_note],
                offset_output=output_dict["offset_output"][:, piano_note],
                offset_shift_output=output_dict["offset_shift_output"][:, piano_note],
                velocity_output=output_dict["velocity_output"][:, piano_note],
                frame_threshold=self.frame_threshold,
            )

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)  # (notes, 5)
        # est_tuples = torch.tensor(est_tuples)
        """(notes, 5), the five columns are onset, offset, onset_shift, 
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes)  # (notes,)
        # est_midi_notes = torch.tensor(est_midi_notes)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]

        est_on_off_note_vels = np.stack(
            (onset_times, offset_times, est_midi_notes, velocities), axis=-1
        )
        # est_on_off_note_vels = torch.stack(
        #     (onset_times, offset_times, est_midi_notes, velocities), dim=-1
        # )
        """(notes, 3), the three columns are onset_times, offset_times and velocity."""

        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)
        # est_on_off_note_vels = est_on_off_note_vels.float()

        return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals.

        Args:
          output_dict: dict, e.g. {
            'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
          est_on_off: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
              [[0.1800, 0.9669],
               [1.1400, 2.6458],
               ...]
        """
        frames_num = output_dict["pedal_frame_output"].shape[0]

        est_tuples = pedal_detection_with_onset_offset_regress(
            frame_output=output_dict["pedal_frame_output"][:, 0],
            offset_output=output_dict["pedal_offset_output"][:, 0],
            offset_shift_output=output_dict["pedal_offset_shift_output"][:, 0],
            frame_threshold=0.5,
        )

        est_tuples = np.array(est_tuples)
        # est_tuples = torch.tensor(est_tuples)
        """(notes, 2), the two columns are pedal onsets and pedal offsets"""

        if len(est_tuples) == 0:
            return np.array([])
            # return torch.tensor([])

        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (
                est_tuples[:, 1] + est_tuples[:, 3]
            ) / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            # est_on_off = torch.stack((onset_times, offset_times), dim=-1)
            est_on_off = est_on_off.astype(np.float32)
            # est_on_off = est_on_off.float()
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times,
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]

        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events: List[NoteData] = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append(
                # {
                #     "onset_time": est_on_off_note_vels[i][0],
                #     "offset_time": est_on_off_note_vels[i][1],
                #     "midi_note": int(est_on_off_note_vels[i][2]),
                #     "velocity": int(est_on_off_note_vels[i][3] * self.velocity_scale),
                # }
                NoteData(
                    onset_time=est_on_off_note_vels[i][0],
                    offset_time=est_on_off_note_vels[i][1],
                    pitch=int(est_on_off_note_vels[i][2]),
                    velocity=int(est_on_off_note_vels[i][3] * self.velocity_scale),
                )
            )

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events.

        Args:
          pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
          offsets. E.g.,
            [[0.1800, 0.9669],
             [1.1400, 2.6458],
             ...]

        Returns:
          pedal_events: list of dict, e.g.,
            [{'onset_time': 0.1800, 'offset_time': 0.9669},
             {'onset_time': 1.1400, 'offset_time': 2.6458},
             ...]
        """
        pedal_events: List[PedalData] = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append(
                # {"onset_time": pedal_on_offs[i, 0], "offset_time": pedal_on_offs[i, 1]}
                PedalData(
                    onset_time=pedal_on_offs[i, 0], offset_time=pedal_on_offs[i, 1]
                )
            )

        return pedal_events


def events_to_midi(
    note_events: List[NoteData],
    pedal_events: Optional[List[PedalData]],
):
    midi = pm.PrettyMIDI()
    piano = pm.Instrument(program=0)

    for note in note_events:
        piano.notes.append(
            pm.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.onset_time,
                end=note.offset_time,
            )
        )

    midi.instruments.append(piano)

    if pedal_events is not None:
        control_change = pm.ControlChange(number=64, value=0, time=0)
        midi.instruments[0].control_changes.append(control_change)

        for pedal in pedal_events:
            midi.instruments[0].control_changes.append(
                pm.ControlChange(number=64, value=127, time=pedal.onset_time)
            )
            midi.instruments[0].control_changes.append(
                pm.ControlChange(number=64, value=0, time=pedal.offset_time)
            )

    return midi
