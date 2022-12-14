"""
KoUniPunc Model
"""
from typing import Optional
import logging

import torch
from torch import Tensor
import torch.nn as nn

from .conv_1d_subsampler import Conv1dSubsampler
from .transformer_fusion_headers import TransformerFusionHeaders
from utils.utils import (
    PUNCTUATION_LABELS,
    W2V_DIM,
    LM_MODEL_CLASSES,
    SM_MODEL_CLASSES,
    get_device,
)


logger = logging.getLogger(__name__)


class KoUniPunc(nn.Module):
    def __init__(self, args):
        super(KoUniPunc, self).__init__()
        # mask_prob: float = 0.1

        self.args = args
        self.device = get_device(args)

        label_lst = PUNCTUATION_LABELS
        self.num_labels = len(label_lst)

        # Lexical Encoder
        self.le_config_class, self.le_model_class, _ = LM_MODEL_CLASSES[
            args.lm_model_type
        ]

        self.le_config = self.le_config_class.from_pretrained(
            args.lm_model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_lst)},
            label2id={label: i for i, label in enumerate(label_lst)},
        )

        self.lexical_encoder = self.le_model_class.from_pretrained(
            args.lm_model_name_or_path, config=self.le_config
        )

        # Acoustic Assistant
        self.mask_wav = args.wav_mask_prob != 0
        self.virtual_embed_dim = args.virtual_embed_dim
        self.sampling_rate = args.wav_sampling_rate

        # Virtual Embedding in Acoustic Assistant
        virtual_embedding = None
        if args.use_virtual:
            virtual_embedding = torch.empty([self.virtual_embed_dim, W2V_DIM])
            virtual_embedding = nn.Parameter(nn.init.xavier_normal_(virtual_embedding))
        self.virtual_embedding = virtual_embedding

        # wav2vec2 model
        (
            self.aa_config_class,
            self.aa_extractor_class,
            self.aa_model_class,
        ) = SM_MODEL_CLASSES[args.sm_model_type]

        self.as_config = self.aa_config_class.from_pretrained(
            args.sm_model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=args.task,
        )

        self.feature_extractor = self.aa_extractor_class.from_pretrained(
            args.sm_model_name_or_path, config=self.as_config
        )

        self.acosutic_assistant = self.aa_model_class.from_pretrained(
            args.sm_model_name_or_path, config=self.as_config
        )

        self.ignore_wav = args.ignore_wav

        self.proj_layer = (
            nn.Linear(in_features=args.w2v_dim, out_features=W2V_DIM)
            if args.w2v_dim != W2V_DIM
            else None
        )

        # Sampler for wav2vec2
        self.subsample = Conv1dSubsampler(
            W2V_DIM,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        # Fusion Header
        self.header_model = TransformerFusionHeaders(
            args.head_hidden_dim,
            args.head_layer_number,
            args.head_cross_layer_number,
            self.num_labels,
            args.head_num,
            args.head_dropout,
        )

    def forward(
        self,
        text_input_ids: Tensor,
        text_attention_mask: Tensor,
        text_token_type_ids: Tensor,
        audio_input: Optional[Tensor] = None,
        audio_length: Optional[Tensor] = None,
        has_audio: bool = False,
    ):
        torch.cuda.empty_cache()

        lexical_encoder_inputs = {
            "input_ids": text_input_ids,
            "attention_mask": text_attention_mask,
        }
        if self.args.lm_model_type != "distilkobert":
            lexical_encoder_inputs["token_type_ids"] = text_token_type_ids

        text_feature = self.lexical_encoder(**lexical_encoder_inputs)
        text_feature = text_feature.last_hidden_state

        # text ??? ???????????? ?????? ?????? ??????
        if self.ignore_wav:
            header_model_input = {
                "text_vec": text_feature,
                "text_vec_mask": text_attention_mask.to(torch.bool),
            }

            logits: Tensor = self.header_model(**header_model_input)

        else:
            bsz, _ = text_input_ids.shape[0], text_input_ids.shape[1]

            # audio_input : (B x max_aud_len)
            # audio feature extraction, padding??? ??? batch????????? max len
            audio_ext_features = self.feature_extractor(
                audio_input,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding="longest",
            )

            # audio_input_values : (B x max_aud_len)
            audio_input_values = audio_ext_features.input_values.view(bsz, -1).to(
                self.device
            )

            audio_features = self.acosutic_assistant(audio_input_values)

            # audio_features : (B x feat_dim x hidden_dim(1024))
            audio_features = audio_features.last_hidden_state

            # W2V_DIM ?????? ?????? projection
            if self.proj_layer:
                # audio_features : (B x feat_dim x W2V_DIM)
                audio_features = self.proj_layer(audio_features)

            # Conv subsampling
            # sampled_audio_features : (B x sampled_dim x W2V_DIM)
            sampled_audio_features, audio_feature_lengths = self.subsample(
                audio_features, audio_length
            )

            # virtual ???????????????
            if self.virtual_embedding is not None:

                has_audio: Tensor = has_audio[0].clone().detach()

                # expanded_virtual_embedding : (B x virtual_embed_dim x W2V_DIM)
                expanded_virtual_embedding = self.virtual_embedding.expand(
                    [bsz, self.virtual_embed_dim, W2V_DIM]
                ).to(sampled_audio_features.dtype)

                # expanded_virtual_embedding_padded : (B x sampled_dim x W2V_DIM)
                expanded_virtual_embedding_padded = torch.zeros_like(
                    sampled_audio_features
                )

                expanded_virtual_embedding_padded[
                    :, : self.virtual_embed_dim, :
                ] = expanded_virtual_embedding

                # audio_masking: (B x sampled_dim x W2V_DIM)
                audio_masking = has_audio.expand(sampled_audio_features.size())

                # sampled_audio_features: (B x sampled_dim x W2V_DIM)
                sampled_audio_features: Tensor = (
                    sampled_audio_features * audio_masking
                    + expanded_virtual_embedding_padded * (~audio_masking)
                )

                # virtual_size:
                # virtual_size: Tensor = self.virtual_embed_dim * torch.ones(
                #     [bsz], dtype=torch.int, device=self.device
                # ) * (~has_audio) + text_length * (has_audio)

                # virtual_size: Tensor = (
                #     self.virtual_embed_dim
                #     * torch.ones([bsz], dtype=torch.int, device=self.device)
                #     * (~has_audio)
                # )

                # temp = (
                #     self.virtual_embed_dim
                #     * torch.ones([bsz], dtype=torch.int)
                #     * ~has_audio
                # )

                # audio_feature_lengths = virtual_size

            # input_mask = (audio_feature_lengths > 1)
            # audio_feature_lengths = audio_feature_lengths * input_mask

            # audio_feature = sampled_audio_features.transpose(0, 1)
            # audio_padding_mask = lengths_to_padding_mask(audio_feature_lengths)

            header_model_input = {
                "text_vec": text_feature,
                "wav_vec": sampled_audio_features,
                "text_vec_mask": text_attention_mask.to(torch.bool),
                # "wav_vec_mask": audio_padding_mask,
            }

            logits: Tensor = self.header_model(**header_model_input)

        return logits
