import copy

from torch import nn

from .. import header_register


class TransformerHeaders(nn.Module):
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

    @classmethod
    def build_model(cls, args, task):
        return cls(
            hidden_size=args.hidden_size,
            layer_number=args.head_layer_number,
            output_size=task.predict_action_number,
            dropout=args.dropout,
            nhead=args.nhead,
        )


@header_register.register("transformer_headers_for_bert_base", TransformerHeaders)
def transformer_headers_for_bert_base(args):
    args = copy.deepcopy(args)
    args.dropout = getattr(args, "dropout", 0.3)
    args.hidden_size = getattr(args, "hidden_size", 768)
    args.head_layer_number = getattr(args, "head_layer_number", 5)
    args.nhead = getattr(args, "nhead", 8)
    return args