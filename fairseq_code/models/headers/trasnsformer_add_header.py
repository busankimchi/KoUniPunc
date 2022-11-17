import copy

from torch import nn

from .transformer_header import TransformerHeaders
from .transformer_cross_header import TransformerCrossHeaders
from .. import header_register


class TransformerAddHeaders(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_number: int,
        output_size: int,
        nhead: int,
        dropout: float,
    ):
        super().__init__()
        self.self_header = TransformerHeaders(
            hidden_size, layer_number, output_size, nhead, dropout
        )
        self.cross_header = TransformerCrossHeaders(
            hidden_size, layer_number, output_size, nhead, dropout
        )

    def forward(self, x, m=None, x_mask=None, m_mask=None):
        self_result = self.self_header(x, x_mask)
        cross_result = self.cross_header(x, m, x_mask, m_mask)
        return self_result + cross_result

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
    "transformer_add_headers_for_bert_base", TransformerAddHeaders
)
def transformer_add_headers_for_bert_base(args):
    args = copy.deepcopy(args)
    args.dropout = getattr(args, "dropout", 0.3)
    args.hidden_size = getattr(args, "hidden_size", 768)
    args.head_layer_number = getattr(args, "head_layer_number", 2)
    args.nhead = getattr(args, "nhead", 8)
    return args
