import copy

from torch import nn

from .. import header_register


class TransformerFullHeaders(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_number: int,
        output_size: int,
        nhead: int,
        dropout: float,
    ):
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_size)