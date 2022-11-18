"""
KoUniPunc Model
"""
import torch
from torch import Tensor
import torch.nn as nn

from ..utils import (
    W2V_DIM,
    PredictResult,
    LM_MODEL_CLASSES,
    SM_MODEL_CLASSES,
    lengths_to_padding_mask,
    load_feature_extractor,
)
from .conv_1d_subsampler import Conv1dSubsampler
from .transformer_fusion_headers import TransformerFusionHeaders


class KoUniPunc(nn.Module):
    def __init__(self, args, label_lst: list):
        # mask_prob: float = 0.1,
        super(KoUniPunc, self).__init__()

        self.args = args
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
        )
        self.lexical_encoder = self.le_model_class.from_pretrained(
            args.lm_model_name_or_path, config=self.le_config
        )

        # Acoustic Assistant
        self.mask_wav = args.wav_mask_prob != 0
        self.virtual_embed_dim = args.virtual_embed_dim

        # Virtual Embedding in Acoustic Assistant
        virtual_embedding = None
        if args.use_virtual:
            virtual_embedding = torch.empty([self.virtual_embed_dim, W2V_DIM])
            virtual_embedding = torch.nn.Parameter(
                nn.init.xavier_normal_(virtual_embedding)
            )
        self.virtual_embedding = virtual_embedding

        # wav2vec2 model
        self.aa_config_class, self.aa_extractor_class = SM_MODEL_CLASSES[
            args.sm_model_type
        ]

        # TODO: 수정 필요
        self.as_config = self.aa_config_class.from_pretrained(
            args.sm_model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task,
            # id2label={str(i): label for i, label in enumerate(label_lst)},
            # label2id={label: i for i, label in enumerate(label_lst)},
        )

        self.feature_extractor = self.aa_extractor_class.from_pretrained(
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
        text_label_ids: Tensor,
        audio_input: Tensor,
        audio_length: Tensor,
        sampling_rate: Tensor,
        text_lengths,
        has_audio=None,
    ):
        torch.cuda.empty_cache()

        lexical_encoder_inputs = {
            "input_ids": text_input_ids,
            "attention_mask": text_attention_mask,
            "labels": text_label_ids,
        }
        if self.args.model_type != "distilkobert":
            lexical_encoder_inputs["token_type_ids"] = text_token_type_ids

        text_feature = self.lexical_encoder(**lexical_encoder_inputs)

        # text 만 사용하는 것과 같은 효과
        if self.ignore_wav:
            output = self.header_model(text_feature)

        else:
            bsz, _ = text_input_ids.shape[0], text_input_ids.shape[1]

            audio_features = self.feature_extractor(
                audio_input,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )

            attention_mask = audio_features.attention_mask

            # audio feature extraction, padding은 각 batch에서의 max len

            # W2V_DIM 차원 으로 projection
            if self.proj_layer:
                audio_input_values = self.proj_layer(audio_features.input_values)

            # Conv subsampling
            audio_input_values, audio_feature_lengths = self.subsample(
                audio_input_values, audio_length
            )
            audio_feature_lengths += (
                audio_input_values.shape[0] - audio_feature_lengths.max()
            )

            # virtual 사용한다면
            if self.virtual_embedding and has_audio:
                has_audio = torch.tensor(has_audio)
                expanded_virtual_embedding = self.virtual_embedding.expand(
                    [bsz, self.virtual_embed_dim, W2V_DIM]
                ).to(audio_input_values.dtype)

                expanded_virtual_embedding_padded = torch.zeros_like(audio_input_values)
                expanded_virtual_embedding_padded[
                    :, : self.virtual_embed_dim, :
                ] = expanded_virtual_embedding

                audio_masking = has_audio.expand([1, 1, 10]).T
                audio_masking = audio_masking.expand(audio_input_values.size())

                audio_input_values: Tensor = (
                    audio_input_values * audio_masking
                    + expanded_virtual_embedding_padded * (~audio_masking)
                )

                virtual_size = self.virtual_embed_dim * torch.ones(
                    [bsz], dtype=torch.int
                ) * (~has_audio) + text_lengths * (has_audio)

                audio_feature_lengths = virtual_size

            # input_mask = (audio_feature_lengths > 1)
            # audio_feature_lengths = audio_feature_lengths * input_mask

            audio_feature = audio_input_values.transpose(0, 1)

            audio_padding_mask = lengths_to_padding_mask(audio_feature_lengths)

            output = self.header_model(
                text_feature, audio_feature, text_attention_mask, audio_padding_mask
            )

        return output, None

    def predict(self, net_output):
        """
        :param net_output: net_output with shape [batch, seq, tag_number] is the output of the forward function.
        :return: A [batch, seq] tensor. The predict result.
        """
        net_output = net_output[0]
        predict_output = net_output.argmax(dim=-1)
        return PredictResult(predict_result=predict_output, predict_logit=net_output)
