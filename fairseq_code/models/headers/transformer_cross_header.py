import copy

from torch import nn

from .. import header_register



class TransformerCrossHeaders(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_number: int,
        output_size: int,
        nhead: int,
        dropout: float,
    ):
        super().__init__()
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

    @classmethod
    def build_model(cls, args, task):
        return cls(
            hidden_size=args.hidden_size,
            layer_number=args.head_layer_number,
            output_size=task.predict_action_number,
            dropout=args.dropout,
            nhead=args.nhead,
        )


@header_register.register(
    "transformer_cross_headers_for_bert_base", TransformerCrossHeaders
)
def transformer_cross_headers_for_bert_base(args):
    args = copy.deepcopy(args)
    args.dropout = getattr(args, "dropout", 0.3)
    args.hidden_size = getattr(args, "hidden_size", 768)
    args.head_layer_number = getattr(args, "head_layer_number", 2)
    args.nhead = getattr(args, "nhead", 8)
    return args
