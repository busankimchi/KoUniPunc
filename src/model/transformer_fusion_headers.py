"""
Transformer fusion header
"""
from typing import Optional
import logging

from torch import nn
from torch import Tensor
from torch.nn import CrossEntropyLoss


logger = logging.getLogger(__name__)


class TransformerCrossHeaders(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_number: int,
        output_size: int,
        nhead: int,
        dropout: float,
    ):
        super(TransformerCrossHeaders, self).__init__()

        self.hidden_size = hidden_size
        one_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout
        )
        self.transformer_head = nn.TransformerDecoder(
            one_layer, num_layers=layer_number
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: Tensor,
        m: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        m_mask: Optional[Tensor] = None,
    ):
        if m is None:
            m = x

        # logger.info(f"TF CROSS :: {x}\t{m}\t{x_mask}\t{m_mask}")
        # logger.info(
        #     f"TF CROSS SIZE :: {x.size()}\t{m.size()}\t{x_mask.size()}\t{m_mask.size()}"
        # )
        x = self.transformer_head(
            x.transpose(0, 1),
            m.transpose(0, 1),
            tgt_key_padding_mask=x_mask,
            memory_key_padding_mask=m_mask,
        ).transpose(0, 1)
        x = self.fc(x)

        return x


class TransformerHeaders(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_number: int,
        output_size: int,
        nhead: int,
        dropout: float,
    ):
        super(TransformerHeaders, self).__init__()

        self.hidden_size = hidden_size
        one_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout
        )
        self.transformer_head = nn.TransformerEncoder(
            one_layer, num_layers=layer_number
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None):
        # logger.info(f"TF HEADER :: X :: {x}\t{x_mask}")

        x = self.transformer_head(
            x.transpose(0, 1), src_key_padding_mask=x_mask
        ).transpose(0, 1)

        # logger.info(f"TF HEADER :: X AFTER HEAD :: {x}")

        x = self.fc(x)

        # logger.info(f"TF HEADER :: X AFTER LINEAR :: {x}\t{x.size()}")

        return x


class TransformerFusionHeaders(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_number: int,
        cross_layer_number: int,
        output_size: int,
        nhead: int,
        dropout: float,
    ):
        super(TransformerFusionHeaders, self).__init__()

        self.num_labels = output_size
        self.self_header = TransformerHeaders(
            hidden_size,
            layer_number,
            output_size,
            nhead,
            dropout,
        )
        self.cross_header = TransformerCrossHeaders(
            hidden_size,
            cross_layer_number,
            output_size,
            nhead,
            dropout,
        )

    def forward(
        self,
        text_vec: Tensor,
        wav_vec: Optional[Tensor] = None,
        text_vec_mask: Optional[Tensor] = None,
        wav_vec_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ):
        self_result = self.self_header(text_vec, text_vec_mask)
        # logger.info(f"TF FUSION :: SELF :: {self_result}\t{self_result.size()}")

        cross_result = self.cross_header(text_vec, wav_vec, text_vec_mask, wav_vec_mask)
        # logger.info(f"TF FUSION :: CROSS :: {cross_result}\t{cross_result.size()}")

        logits: Tensor = self_result + cross_result
        # logger.info(f"LOGITS ::: {logits}\t{logits.size()}")

        # logger.info(
        #     f"LOGIT ALL :: {logits.view(-1, self.num_labels)}, {logits.view(-1, self.num_labels).size()}"
        # )
        # logger.info(f"LABEL ALL :: {labels.view(-1)}")

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # logger.info(f"LOSS ::: {loss}")

        return loss, logits
