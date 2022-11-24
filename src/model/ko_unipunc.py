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
from ..utils import (
    PUNCTUATION_LABELS,
    W2V_DIM,
    LM_MODEL_CLASSES,
    SM_MODEL_CLASSES,
    get_device,
    lengths_to_padding_mask,
)


logger = logging.getLogger(__name__)


class KoUniPunc(nn.Module):
    def __init__(self, args):
        super(KoUniPunc, self).__init__()
        # mask_prob: float = 0.1

        self.args = args
        self.device = get_device(args)

        label_lst = PUNCTUATION_LABELS
        num_labels = len(label_lst)

        # Lexical Encoder
        self.le_config_class, self.le_model_class, _ = LM_MODEL_CLASSES[
            args.lm_model_type
        ]

        # TODO: 수정 필요
        self.le_config = self.le_config_class.from_pretrained(
            args.lm_model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_lst)},
            label2id={label: i for i, label in enumerate(label_lst)},
            # max_position_embeddings=1024,
        )

        # logger.info(f"LE CONFIG :: {self.le_config}")

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
            args.sm_model_name_or_path, num_labels=num_labels, finetuning_task=args.task
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
            num_labels,
            args.head_num,
            args.head_dropout,
        )

    def forward(
        self,
        text_input_ids: Tensor,
        text_attention_mask: Tensor,
        text_token_type_ids: Tensor,
        labels: Tensor,
        text_length: Tensor,
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

        # logger.info(f"LEXICAL :: {lexical_encoder_inputs}")

        text_feature = self.lexical_encoder(**lexical_encoder_inputs)
        text_feature = text_feature.last_hidden_state

        # logger.info(f"TEXT FEATURE :: {text_feature}")

        # text 만 사용하는 것과 같은 효과
        if self.ignore_wav:
            loss, logits = self.header_model(text_feature)

        else:
            bsz, _ = text_input_ids.shape[0], text_input_ids.shape[1]

            # audio_input : (B x max_aud_len)
            # logger.info(f"AUD INPUT SIZE ::: {audio_input.size()}")

            # audio feature extraction, padding은 각 batch에서의 max len
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

            # logger.info(f"AUDIO :: {audio_input_values}")
            # logger.info(f"AUDIO SIZE :: {audio_input_values.size()}")

            audio_features = self.acosutic_assistant(audio_input_values)

            # audio_features : (B x feat_dim x hidden_dim(1024))
            audio_features = audio_features.last_hidden_state

            # logger.info(f"AUDIO FEATURE :: {audio_features}")
            # logger.info(f"AUDIO FEATURE SIZE :: {audio_features.size()}")

            # W2V_DIM 차원 으로 projection
            if self.proj_layer:
                # audio_features : (B x feat_dim x W2V_DIM)
                audio_features = self.proj_layer(audio_features)

            # Conv subsampling
            # sampled_audio_features : (B x sampled_dim x W2V_DIM)
            sampled_audio_features, audio_feature_lengths = self.subsample(
                audio_features, audio_length
            )

            # logger.info(
            #     f"SAMPLED AUD FEAT LENS :: {audio_feature_lengths}\t{sampled_audio_features.shape}"
            # )

            # audio_feature_lengths += (
            #     sampled_audio_features.shape[0] - audio_feature_lengths.max()
            # )

            # logger.info(
            #     f"SAMPLED AUD FEAT :: {sampled_audio_features}, {audio_feature_lengths}"
            # )
            # logger.info(
            #     f"SAMPLED AUD FEAT SIZE :: {sampled_audio_features.size()}, {audio_feature_lengths.size()}"
            # )

            # logger.info(f"HAS AUDIO :: {has_audio}")
            # logger.info(f"VIRT EMBED :: {self.virtual_embedding}")
            # logger.info(f"VIRT EMBED SIZE :: {self.virtual_embedding.size()}")

            # virtual 사용한다면
            if self.virtual_embedding is not None:
                # has_audio = torch.tensor(has_audio)
                has_audio: Tensor = has_audio.clone().detach()

                # expanded_virtual_embedding : (B x virtual_embed_dim x W2V_DIM)
                expanded_virtual_embedding = self.virtual_embedding.expand(
                    [bsz, self.virtual_embed_dim, W2V_DIM]
                ).to(sampled_audio_features.dtype)

                # logger.info(f"EXP VIRT EMBED :: {expanded_virtual_embedding}")
                # logger.info(
                #     f"EXP VIRT EMBED SIZE :: {expanded_virtual_embedding.size()}"
                # )

                # expanded_virtual_embedding_padded : (B x sampled_dim x W2V_DIM)
                expanded_virtual_embedding_padded = torch.zeros_like(
                    sampled_audio_features
                )

                # logger.info(f"VIRT EMBED PAD :: {expanded_virtual_embedding_padded}")
                # logger.info(
                #     f"VIRT EMBED PAD SIZE:: {expanded_virtual_embedding_padded.size()}"
                # )

                expanded_virtual_embedding_padded[
                    :, : self.virtual_embed_dim, :
                ] = expanded_virtual_embedding

                # logger.info(
                #     f"APPLIED VIRT EMBED PAD :: {expanded_virtual_embedding_padded}"
                # )
                # logger.info(
                #     f"APPLIED VIRT EMBED PAD SIZE:: {expanded_virtual_embedding_padded.size()}"
                # )

                # audio_masking = has_audio.expand([1, 1, 10]).T
                # audio_masking = audio_masking.expand(sampled_audio_features.size())

                # audio_masking: (B x sampled_dim x W2V_DIM)
                audio_masking = has_audio.expand(sampled_audio_features.size())

                # logger.info(f"AUD MASING :: {audio_masking}")
                # logger.info(f"AUD MASING SIZE :: {audio_masking.size()}")

                # sampled_audio_features: (B x sampled_dim x W2V_DIM)
                sampled_audio_features: Tensor = (
                    sampled_audio_features * audio_masking
                    + expanded_virtual_embedding_padded * (~audio_masking)
                )

                # logger.info(f"SAMPLED AUD FEAT :: {sampled_audio_features}")
                # logger.info(f"SAMPLED AUD FEAT SIZE :: {sampled_audio_features.size()}")

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

                # logger.info(
                #     f"VIRTUAL TENSOR :: {temp}\t{temp.size()}",
                #     f"NOT AUDIO :: {~has_audio}",
                #     f"TEXT TENSOR :: {text_length * has_audio}",
                # )

                # logger.info(f"VIR SIZE :: {virtual_size}")

                # audio_feature_lengths = virtual_size

                # logger.info(f"FINAL AUDIO FEAT LENS :: {audio_feature_lengths}")
                # logger.info(
                #     f"FINAL AUDIO FEAT LENS SIZE :: {audio_feature_lengths.size()}"
                # )

            # input_mask = (audio_feature_lengths > 1)
            # audio_feature_lengths = audio_feature_lengths * input_mask

            # audio_feature = sampled_audio_features.transpose(0, 1)

            audio_padding_mask = lengths_to_padding_mask(audio_feature_lengths)

            # logger.info(f"AUD PADDING MASK :: {audio_padding_mask}")

            header_model_input = {
                "text_vec": text_feature,
                "wav_vec": sampled_audio_features,
                "text_vec_mask": text_attention_mask.to(torch.bool),
                # "wav_vec_mask": audio_padding_mask,
                "labels": labels,
            }

            loss, logits = self.header_model(**header_model_input)

        return loss, logits
