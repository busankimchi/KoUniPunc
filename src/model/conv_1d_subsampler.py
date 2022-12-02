from typing import List, Optional
import logging

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 3],
    ):
        super(Conv1dSubsampler, self).__init__()

        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=8,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def _get_out_seq_lens_tensor(self, in_seq_lens_tensor: Tensor) -> Optional[Tensor]:
        if in_seq_lens_tensor is None:
            return None

        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 8 + 1).floor().long()

        return out

    def forward(self, src_tokens: Tensor, src_lengths: Tensor):
        # B x T x (C x D)

        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T

        for conv in self.conv_layers:
            x = conv(x)
            x = F.glu(x, dim=1)

        x = x.transpose(1, 2).contiguous()  # -> B x T x (C x D)

        return x, self._get_out_seq_lens_tensor(src_lengths)
