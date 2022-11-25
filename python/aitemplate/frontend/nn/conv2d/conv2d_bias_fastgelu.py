"""
conv bias fastgelu module
"""
from .common_conv2d_bias_act import Conv2dBiasAct


class Conv2dBiasFastGelu(Conv2dBiasAct):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        dtype="float16",
    ):
        super().__init__(
            "conv2d_bias_fastgelu",
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dtype,
        )
