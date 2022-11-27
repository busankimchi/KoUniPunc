from dataclasses import dataclass
from typing import List, Optional
import logging

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio


logger = logging.getLogger(__name__)


@dataclass
class InputFeature:
    """A single set of features of data."""

    text_input_ids: list
    text_attention_mask: list
    labels: list
    text_token_type_ids: Optional[list] = None

    audio_path: Optional[str] = None


class WelfareCallDataset(Dataset):
    def __init__(self, args, features: List[InputFeature]):
        self.args = args
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]

        text_input_ids = torch.tensor(feature.text_input_ids, dtype=torch.long)
        text_attention_mask = torch.tensor(
            feature.text_attention_mask, dtype=torch.long
        )
        text_token_type_ids = torch.tensor(
            feature.text_token_type_ids, dtype=torch.long
        )
        labels = torch.tensor(feature.labels, dtype=torch.long)

        # transform audio features
        has_audio, audio_input, audio_length = False, None, 0
        if feature.audio_path is not None:
            has_audio = True
            speech_array, sampling_rate = torchaudio.load(feature.audio_path)

            if self.args.wav_sampling_rate != sampling_rate:
                speech_array = torchaudio.functional.resample(
                    speech_array,
                    orig_freq=sampling_rate,
                    new_freq=self.args.wav_sampling_rate,
                )

            audio_input = speech_array[0]
            speech_array_length = len(audio_input)

            padding_length = self.args.max_aud_len - speech_array_length

            if padding_length > 0:
                audio_input = F.pad(
                    audio_input, pad=(0, padding_length), mode="constant", value=0
                )

            audio_length = torch.tensor(speech_array_length, dtype=torch.long)

        return (
            text_input_ids,
            text_attention_mask,
            text_token_type_ids,
            audio_input,
            audio_length,
            has_audio,
            labels,
        )
