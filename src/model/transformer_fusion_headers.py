"""
Transformer fusion header
"""
from torch import nn


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

    def forward(self, x, m=None, x_mask=None, m_mask=None):
        if m is None:
            m = x

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

    def forward(self, x, x_mask=None):
        x = self.transformer_head(
            x.transpose(0, 1), src_key_padding_mask=x_mask
        ).transpose(0, 1)
        x = self.fc(x)

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

    def forward(self, text_vec, wav_vec=None, text_vec_mask=None, wav_vec_mask=None):
        self_result = self.self_header(text_vec, text_vec_mask)
        cross_result = self.cross_header(text_vec, wav_vec, text_vec_mask, wav_vec_mask)

        return self_result + cross_result
