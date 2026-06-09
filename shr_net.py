import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, List, Dict, Union, Optional


def _to_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)
    if isinstance(x, (list, tuple)):
        assert len(x) == 3, f"Expected 3 elements, got {x}"
        return tuple(x)
    raise TypeError(f"Unsupported type: {type(x)}")


def _get_same_padding(kernel_size, dilation=1):
    kernel_size = _to_3tuple(kernel_size)
    dilation = _to_3tuple(dilation)
    return tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))


def get_norm_layer(norm: str, num_channels: int):
    norm = norm.lower()

    if norm == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)

    if norm == "batch":
        return nn.BatchNorm3d(num_channels)

    if norm == "group":
        groups = 8
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)

    if norm == "none":
        return nn.Identity()

    raise ValueError(f"Unsupported norm type: {norm}")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(1, 3, 3),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        norm: str = "instance",
        act: bool = True,
    ):
        super().__init__()

        kernel_size = _to_3tuple(kernel_size)
        stride = _to_3tuple(stride)
        dilation = _to_3tuple(dilation)
        padding = _get_same_padding(kernel_size, dilation)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = get_norm_layer(norm, out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SEBlock3D(nn.Module):
    """
    轻量通道注意力。
    作用：增强出血相关通道，抑制无关背景响应。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class AnisoResBlock(nn.Module):
    """
    各向异性残差块。

    针对你的任务：
    1. 前期用 kernel=(1,3,3)，主要学习层内出血形态；
    2. 中后期可以加入 pseudo-3D: (1,3,3) + (3,1,1)，轻量学习 z 方向上下文；
    3. 残差结构缓解 loss 震荡。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,         kernel_size=(1, 3, 3),
        stride=(1, 1, 1),
        norm: str = "instance",
        use_se: bool = True,
        pseudo_3d: bool = False,
        disable_residual_shortcut: bool = False,
    ):
        super().__init__()
        self.disable_residual_shortcut = disable_residual_shortcut

        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm,
            act=True,
        )

        if pseudo_3d:
            self.conv_z = ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                norm=norm,
                act=True,
            )
        else:
            self.conv_z = nn.Identity()

        self.conv2 = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            norm=norm,
            act=False,
        )

        self.se = SEBlock3D(out_channels) if use_se else nn.Identity()

        if in_channels != out_channels or _to_3tuple(stride) != (1, 1, 1):
            self.shortcut = ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=(1, 1, 1),
                stride=stride,
                norm=norm,
                act=False,
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv_z(out)
        out = self.conv2(out)
        out = self.se(out)

        if not self.disable_residual_shortcut:
            out = out + identity
        out = self.act(out)

        return out


class StandardResBlock(nn.Module):
    """
    Standard isotropic residual block used when ARC is disabled for ablations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=(1, 1, 1),
        norm: str = "instance",
        use_se: bool = True,
    ):
        super().__init__()

        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            norm=norm,
            act=True,
        )
        self.conv2 = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            norm=norm,
            act=False,
        )
        self.se = SEBlock3D(out_channels) if use_se else nn.Identity()

        if in_channels != out_channels or _to_3tuple(stride) != (1, 1, 1):
            self.shortcut = ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=(1, 1, 1),
                stride=stride,
                norm=norm,
                act=False,
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return self.act(out + identity)


class DownBlock(nn.Module):
    """
    下采样模块。
    关键：前两层只在 xy 方向下采样，不压缩 z。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=(1, 2, 2),
        kernel_size=(1, 3, 3),
        norm: str = "instance",
        pseudo_3d: bool = False,
        use_se: bool = True,
        use_arc: bool = True,
        arc_no_shortcut: bool = False,
    ):
        super().__init__()
        block_cls = AnisoResBlock if use_arc else StandardResBlock

        if use_arc:
            self.block1 = block_cls(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm=norm,
                use_se=use_se,
                pseudo_3d=pseudo_3d,
                disable_residual_shortcut=arc_no_shortcut,
            )
            self.block2 = block_cls(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=(1, 1, 1),
                norm=norm,
                use_se=use_se,
                pseudo_3d=pseudo_3d,
                disable_residual_shortcut=arc_no_shortcut,
            )
        else:
            self.block1 = block_cls(
                in_channels,
                out_channels,
                stride=stride,
                norm=norm,
                use_se=use_se,
            )
            self.block2 = block_cls(
                out_channels,
                out_channels,
                stride=(1, 1, 1),
                norm=norm,
                use_se=use_se,
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class HMCModule(nn.Module):
    """
    Hemorrhage Multi-scale Context Module.

    针对混合脑出血形态：
    1. 小出血：局部卷积分支；
    2. 大出血：更大感受野；
    3. 条带状/脑沟/脑室内出血：层内 dilation；
    4. 厚层 CT：限制 z 方向 dilation，不盲目扩张 z。
    """

    def __init__(
        self,
        channels: int,
        norm: str = "instance",
        use_se: bool = True,
        use_dilation: bool = True,
        use_3d_branch: bool = True,
    ):
        super().__init__()

        branch_channels = max(channels // 4, 16)
        self.branch_channels = branch_channels

        self.branch1 = ConvNormAct(
            channels,
            branch_channels,
            kernel_size=(1, 1, 1),
            dilation=(1, 1, 1),
            norm=norm,
        )

        self.branch2 = ConvNormAct(
            channels,
            branch_channels,
            kernel_size=(1, 3, 3),
            dilation=(1, 1, 1),
            norm=norm,
        )

        self.branch3 = ConvNormAct(
            channels,
            branch_channels,
            kernel_size=(1, 3, 3),
            dilation=(1, 2, 2) if use_dilation else (1, 1, 1),
            norm=norm,
        )

        if use_3d_branch:
            self.branch4 = ConvNormAct(
                channels,
                branch_channels,
                kernel_size=(3, 3, 3),
                dilation=(1, 2, 2) if use_dilation else (1, 1, 1),
                norm=norm,
            )
        else:
            self.branch4 = None

        self.fuse = nn.Sequential(
            ConvNormAct(
                branch_channels * 4,
                channels,
                kernel_size=(1, 1, 1),
                norm=norm,
                act=True,
            ),
            AnisoResBlock(
                channels,
                channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                norm=norm,
                use_se=use_se,
                pseudo_3d=False,
            ),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        if self.branch4 is None:
            b4 = torch.zeros(
                x.shape[0],
                self.branch_channels,
                x.shape[2],
                x.shape[3],
                x.shape[4],
                device=x.device,
                dtype=x.dtype,
            )
        else:
            b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fuse(out)
        return out


class AttentionGate3D(nn.Module):
    """
    注意力跳跃连接。

    作用：
    1. 保留出血相关的浅层边界信息；
    2. 抑制颅骨、钙化、伪影等高密度干扰；
    3. 减少 false positive。
    """

    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        inter_channels: Optional[int] = None,
        norm: str = "instance",
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)

        self.theta_x = ConvNormAct(
            skip_channels,
            inter_channels,
            kernel_size=(1, 1, 1),
            norm=norm,
            act=False,
        )

        self.phi_g = ConvNormAct(
            gate_channels,
            inter_channels,
            kernel_size=(1, 1, 1),
            norm=norm,
            act=False,
        )

        self.psi = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip, gate):
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(
                gate,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        theta = self.theta_x(skip)
        phi = self.phi_g(gate)

        attn = self.psi(theta + phi)
        return skip * attn


class SimpleAttentionGate3D(nn.Module):
    """
    Skip-only attention used for component-level AG ablations.
    """

    def __init__(self, skip_channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(skip_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip, gate=None):
        return skip * self.attn(skip)


class UpBlock(nn.Module):
    """
    解码器上采样模块。
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size=(1, 3, 3),
        norm: str = "instance",
        pseudo_3d: bool = False,
        use_attention: bool = True,
        attention_simple: bool = False,
        use_se: bool = True,
        use_arc: bool = True,
        arc_no_shortcut: bool = False,
    ):
        super().__init__()
        block_cls = AnisoResBlock if use_arc else StandardResBlock

        self.up_proj = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=(1, 1, 1),
            norm=norm,
            act=True,
        )

        if use_attention and attention_simple:
            self.attention = SimpleAttentionGate3D(skip_channels)
        elif use_attention:
            self.attention = AttentionGate3D(skip_channels, out_channels, norm=norm)
        else:
            self.attention = nn.Identity()

        if use_arc:
            self.fuse = nn.Sequential(
                block_cls(
                    out_channels + skip_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                    pseudo_3d=pseudo_3d,
                    disable_residual_shortcut=arc_no_shortcut,
                ),
                block_cls(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                    pseudo_3d=pseudo_3d,
                    disable_residual_shortcut=arc_no_shortcut,
                ),
            )
        else:
            self.fuse = nn.Sequential(
                block_cls(
                    out_channels + skip_channels,
                    out_channels,
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                ),
                block_cls(
                    out_channels,
                    out_channels,
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                ),
            )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        x = self.up_proj(x)

        if isinstance(self.attention, (AttentionGate3D, SimpleAttentionGate3D)):
            skip = self.attention(skip, x)

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x


class SmallBleedRefinementHead(nn.Module):
    """
    小出血细化头。

    融合：
    1. decoder 最后一层语义特征；
    2. stem 浅层高分辨率细节特征。

    目的：
    1. 提升小出血召回；
    2. 减少 Dice=0 病例；
    3. 改善边界和小区域断裂。
    """

    def __init__(
        self,
        decoder_channels: int,
        shallow_channels: int,
        out_channels: int,
        norm: str = "instance",
        use_se: bool = True,
        use_arc: bool = True,
        no_shallow: bool = False,
        simple_refine: bool = False,
        arc_no_shortcut: bool = False,
    ):
        super().__init__()
        block_cls = AnisoResBlock if use_arc else StandardResBlock

        self.no_shallow = no_shallow
        in_channels = decoder_channels if no_shallow else decoder_channels + shallow_channels

        if simple_refine:
            self.refine = nn.Sequential(
                ConvNormAct(
                    in_channels,
                    decoder_channels,
                    kernel_size=(1, 3, 3),
                    norm=norm,
                    act=True,
                ),
                nn.Conv3d(decoder_channels, out_channels, kernel_size=1),
            )
        elif use_arc:
            self.refine = nn.Sequential(
                block_cls(
                    in_channels,
                    decoder_channels,
                    kernel_size=(1, 3, 3),
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                    pseudo_3d=False,
                    disable_residual_shortcut=arc_no_shortcut,
                ),
                ConvNormAct(
                    decoder_channels,
                    decoder_channels,
                    kernel_size=(1, 3, 3),
                    norm=norm,
                    act=True,
                ),
                nn.Conv3d(decoder_channels, out_channels, kernel_size=1),
            )
        else:
            self.refine = nn.Sequential(
                block_cls(
                    in_channels,
                    decoder_channels,
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                ),
                ConvNormAct(
                    decoder_channels,
                    decoder_channels,
                    kernel_size=(3, 3, 3),
                    norm=norm,
                    act=True,
                ),
                nn.Conv3d(decoder_channels, out_channels, kernel_size=1),
            )

    def forward(self, decoder_feature, shallow_feature):
        if not self.no_shallow and decoder_feature.shape[2:] != shallow_feature.shape[2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=shallow_feature.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        if self.no_shallow:
            x = decoder_feature
        else:
            x = torch.cat([decoder_feature, shallow_feature], dim=1)
        x = self.refine(x)
        return x


class SegHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.head = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.head(x)


class ForegroundPresenceHead(nn.Module):
    """
    前景存在辅助头。

    用于缓解小出血全背景预测问题。
    训练时判断当前 patch/case 是否存在出血。
    推理时可以不用。
    """

    def __init__(self, in_channels: int):
        super().__init__()

        hidden = max(in_channels // 4, 32)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.head(x)


class AnisoHemorrhageUNet(nn.Module):
    """
    Aniso-HemorrhageUNet

    面向厚层 CT 脑出血分割的各向异性小目标敏感网络。

    主要针对：
    1. CT 层厚 5-6mm，z 方向分辨率低；
    2. 小出血容易漏检；
    3. 数据包含不同类型脑出血，形态差异大；
    4. 训练 loss 震荡；
    5. 部分病例 Dice 接近 0。

    输入:
        x: [B, C, D, H, W]

    输出:
        如果 return_dict=True:
            {
                "logits": main segmentation logits,
                "aux_logits": [aux1, aux2],
                "presence_logit": foreground presence logit
            }

        如果 return_dict=False:
            main segmentation logits
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Sequence[int] = (32, 64, 128, 256, 320),
        norm: str = "instance",
        deep_supervision: bool = True,
        presence_head: bool = True,
        use_attention: bool = True,
        use_hmc: bool = True,
        use_small_refine: bool = True,
        use_se: bool = True,
        use_pseudo3d: bool = True,
        use_arc: bool = True,
        arc_no_shortcut: bool = False,
        hmc_no_dilation: bool = False,
        hmc_no_3d_branch: bool = False,
        attention_simple: bool = False,
        sbr_no_shallow: bool = False,
        sbr_simple: bool = False,
        return_dict: bool = True,
    ):
        super().__init__()

        assert len(channels) == 5, "channels should have 5 stages, e.g. (32,64,128,256,320)"

        c0, c1, c2, c3, c4 = channels

        self.deep_supervision = deep_supervision
        self.use_presence_head = presence_head
        self.use_small_refine = use_small_refine
        self.return_dict = return_dict
        block_cls = AnisoResBlock if use_arc else StandardResBlock

        # Stem: 不下采样，只提取层内细节
        if use_arc:
            self.stem = nn.Sequential(
                block_cls(
                    in_channels,
                    c0,
                    kernel_size=(1, 3, 3),
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                    pseudo_3d=False,
                    disable_residual_shortcut=arc_no_shortcut,
                ),
                block_cls(
                    c0,
                    c0,
                    kernel_size=(1, 3, 3),
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                    pseudo_3d=False,
                    disable_residual_shortcut=arc_no_shortcut,
                ),
            )
        else:
            self.stem = nn.Sequential(
                block_cls(
                    in_channels,
                    c0,
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                ),
                block_cls(
                    c0,
                    c0,
                    stride=(1, 1, 1),
                    norm=norm,
                    use_se=use_se,
                ),
            )

        # Encoder
        # 前两层只压缩 xy，不压缩 z
        self.enc1 = DownBlock(
            c0,
            c1,
            stride=(1, 2, 2),
            kernel_size=(1, 3, 3),
            norm=norm,
            pseudo_3d=False,
            use_se=use_se,
            use_arc=use_arc,
            arc_no_shortcut=arc_no_shortcut,
        )

        self.enc2 = DownBlock(
            c1,
            c2,
            stride=(1, 2, 2),
            kernel_size=(1, 3, 3),
            norm=norm,
            pseudo_3d=use_pseudo3d,
            use_se=use_se,
            use_arc=use_arc,
            arc_no_shortcut=arc_no_shortcut,
        )

        # 中后层开始适度融合 z 方向信息
        self.enc3 = DownBlock(
            c2,
            c3,
            stride=(2, 2, 2),
            kernel_size=(3, 3, 3),
            norm=norm,
            pseudo_3d=False,
            use_se=use_se,
            use_arc=use_arc,
            arc_no_shortcut=arc_no_shortcut,
        )

        # 最后一层继续扩大 xy 感受野，但不再压缩 z
        self.enc4 = DownBlock(
            c3,
            c4,
            stride=(1, 2, 2),
            kernel_size=(3, 3, 3),
            norm=norm,
            pseudo_3d=False,
            use_se=use_se,
            use_arc=use_arc,
            arc_no_shortcut=arc_no_shortcut,
        )

        # Bottleneck 多尺度出血上下文
        self.hmc = (
            HMCModule(
                c4,
                norm=norm,
                use_se=use_se,
                use_dilation=not hmc_no_dilation,
                use_3d_branch=not hmc_no_3d_branch,
            )
            if use_hmc
            else nn.Identity()
        )

        # Decoder
        self.dec3 = UpBlock(
            in_channels=c4,
            skip_channels=c3,
            out_channels=c3,
            kernel_size=(3, 3, 3),
            norm=norm,
            pseudo_3d=False,
            use_attention=use_attention,
            attention_simple=attention_simple,
            use_se=use_se,
            use_arc=use_arc,
            arc_no_shortcut=arc_no_shortcut,
        )

        self.dec2 = UpBlock(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2,
            kernel_size=(1, 3, 3),
            norm=norm,
            pseudo_3d=use_pseudo3d,
            use_attention=use_attention,
            attention_simple=attention_simple,
            use_se=use_se,
            use_arc=use_arc,
            arc_no_shortcut=arc_no_shortcut,
        )

        self.dec1 = UpBlock(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1,
            kernel_size=(1, 3, 3),
            norm=norm,
            pseudo_3d=False,
            use_attention=use_attention,
            attention_simple=attention_simple,
            use_se=use_se,
            use_arc=use_arc,
            arc_no_shortcut=arc_no_shortcut,
        )

        self.dec0 = UpBlock(
            in_channels=c1,
            skip_channels=c0,
            out_channels=c0,
            kernel_size=(1, 3, 3),
            norm=norm,
            pseudo_3d=False,
            use_attention=use_attention,
            attention_simple=attention_simple,
            use_se=use_se,
            use_arc=use_arc,
            arc_no_shortcut=arc_no_shortcut,
        )

        # 小出血细化头作为最终输出
        if self.use_small_refine:
            self.final_head = SmallBleedRefinementHead(
                decoder_channels=c0,
                shallow_channels=c0,
                out_channels=out_channels,
                norm=norm,
                use_se=use_se,
                use_arc=use_arc,
                no_shallow=sbr_no_shallow,
                simple_refine=sbr_simple,
                arc_no_shortcut=arc_no_shortcut,
            )
        else:
            self.final_head = SegHead(c0, out_channels)

        # 深监督输出
        if self.deep_supervision:
            self.aux_head1 = SegHead(c1, out_channels)
            self.aux_head2 = SegHead(c2, out_channels)
            self.aux_head3 = SegHead(c3, out_channels)

        # 前景存在辅助头
        if self.use_presence_head:
            self.presence = ForegroundPresenceHead(c4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e0 = self.stem(x)     # [B, c0, D, H, W]
        e1 = self.enc1(e0)    # [B, c1, D, H/2, W/2]
        e2 = self.enc2(e1)    # [B, c2, D, H/4, W/4]
        e3 = self.enc3(e2)    # [B, c3, D/2, H/8, W/8]
        e4 = self.enc4(e3)    # [B, c4, D/2, H/16, W/16]

        # Context
        b = self.hmc(e4)

        # Decoder
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        # Main output
        if self.use_small_refine:
            logits = self.final_head(d0, e0)
        else:
            logits = self.final_head(d0)

        if not self.return_dict:
            return logits

        output = {
            "logits": logits,
        }

        if self.deep_supervision:
            output["aux_logits"] = [
                self.aux_head1(d1),
                self.aux_head2(d2),
                self.aux_head3(d3),
            ]

        if self.use_presence_head:
            output["presence_logit"] = self.presence(b)

        return output


def build_aniso_hemorrhage_unet(
    in_channels: int = 1,
    out_channels: int = 2,
    base_channels: int = 32,
    norm: str = "instance",
    deep_supervision: bool = True,
    presence_head: bool = True,
    use_attention: bool = True,
    use_hmc: bool = True,
    use_small_refine: bool = True,
    use_se: bool = True,
    use_pseudo3d: bool = True,
    use_arc: bool = True,
    arc_no_shortcut: bool = False,
    hmc_no_dilation: bool = False,
    hmc_no_3d_branch: bool = False,
    attention_simple: bool = False,
    sbr_no_shallow: bool = False,
    sbr_simple: bool = False,
    return_dict: bool = True,
):
    channels = (
        base_channels,
        base_channels * 2,
        base_channels * 4,
        base_channels * 8,
        base_channels * 10,
    )

    model = AnisoHemorrhageUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        norm=norm,
        deep_supervision=deep_supervision,
        presence_head=presence_head,
        use_attention=use_attention,
        use_hmc=use_hmc,
        use_small_refine=use_small_refine,
        use_se=use_se,
        use_pseudo3d=use_pseudo3d,
        use_arc=use_arc,
        arc_no_shortcut=arc_no_shortcut,
        hmc_no_dilation=hmc_no_dilation,
        hmc_no_3d_branch=hmc_no_3d_branch,
        attention_simple=attention_simple,
        sbr_no_shallow=sbr_no_shallow,
        sbr_simple=sbr_simple,
        return_dict=return_dict,
    )

    return model


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_aniso_hemorrhage_unet(
        in_channels=1,
        out_channels=2,
        base_channels=32,
        deep_supervision=True,
        presence_head=True,
        return_dict=True,
    ).to(device)

    x = torch.randn(1, 1, 32, 192, 192).to(device)

    with torch.no_grad():
        y = model(x)

    print("Main logits:", y["logits"].shape)

    if "aux_logits" in y:
        for i, aux in enumerate(y["aux_logits"]):
            print(f"Aux {i}:", aux.shape)

    if "presence_logit" in y:
        print("Presence:", y["presence_logit"].shape)

    total, trainable = count_parameters(model)
    print(f"Total params: {total / 1e6:.2f} M")
    print(f"Trainable params: {trainable / 1e6:.2f} M")
