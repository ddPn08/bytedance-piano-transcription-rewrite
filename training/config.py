from pydantic import BaseModel


class FeatureConfig(BaseModel):
    sampling_rate: int = 16000
    win_length: int = 2048
    num_mels: int = 229
    f_min: int = 30
    center: bool = True
    pad_mode: str = "reflect"
    ref: float = 1.0
    amin: float = 1e-10


class MidiConfig(BaseModel):
    num_notes: int = 88
    begin_note: int = 21
    velocity_scale: int = 128


class Config(BaseModel):
    feature: FeatureConfig = FeatureConfig()
    midi: MidiConfig = MidiConfig()
    hop_seconds: float = 1.0
    frames_per_second: int = 100
    segment_seconds: float = 10.0
