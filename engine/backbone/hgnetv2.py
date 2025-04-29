"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .common import FrozenBatchNorm2d     # 自定义的冻结BN层
from ..core import register               # 模型注册装饰器
import logging

# 初始化常量定义
kaiming_normal_ = nn.init.kaiming_normal_  # Kaiming正态分布初始化
zeros_ = nn.init.zeros_      # 零初始化
ones_ = nn.init.ones_        # 一初始化

__all__ = ['HGNetv2']        # 模块导出列表


class LearnableAffineBlock(nn.Module):
    """
    可学习的仿射变换模块
    关键点：
        - 学习缩放和偏移参数
        - 增强网络非线性表达能力
        - 可用于替代传统激活函数
    """
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True) # 可学习缩放因子
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)   # 可学习偏移量

    def forward(self, x):
        return self.scale * x + self.bias       # 应用仿射变换


class ConvBNAct(nn.Module):
    """基础卷积模块（Conv + BN + Activation）
    关键参数：
        - padding: 'same' 时使用补零策略保持特征图尺寸
        - use_lab: 是否使用可学习仿射变换
    结构说明：
        |--Conv2d--BN--ReLU--Lab--|
    """
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1,
                 padding='', use_act=True, use_lab=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab

        # 处理padding模式
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),  # 右下各补1像素
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,  # 自动计算padding
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()      # 当use_act=True时候使用ReLU激活函数，否则使用Identity()占位
        else:
            self.act = nn.Identity()  # nn.Identity()是一个占位符，不做任何计算，用于保持层数
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock() # 当use_act和use_lab都为True时候，使用可学习仿射变换
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        # |--Conv2d--BN--ReLU--Lab--|
        # |--Conv2d--BN--Identity--Identity--|
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    """轻量级卷积模块
    结构说明：
        |--1x1 Conv--Depthwise Conv--|
    关键点：
        - 使用深度可分离卷积减少计算量
        - 第一个卷积用于通道调整，第二个进行空间特征提取
    """
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    """网络入口模块（初步特征提取）
    结构流程：
        1. 3x3 Conv stride2下采样
        2. 并行支路处理
        3. 特征拼接
        4. 进一步下采样和通道调整
    """
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct( in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab,
        )
        self.stem2a = ConvBNAct( mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab,
        )
        self.stem2b = ConvBNAct( mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab,
        )
        self.stem3 = ConvBNAct( mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab,
        )
        self.stem4 = ConvBNAct( mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))     # 右下补零
        x2 = self.stem2a(x)            # 并行支路处理x2
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)              # 并行支路处理x1
        x = torch.cat([x1, x2], dim=1) # 拼接x1和x2
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    """ESE (Extremely Simple ECA) 注意力模块
    关键点：
        - 通过全局平均池化获取通道注意力
        - 使用1x1卷积学习通道相关性
        - 参数量极小但效果显著
    """
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d( chs, chs, kernel_size=1, stride=1, padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)   # 空间维度平均池化，在(H,W)维度上取平均
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)      # 通道注意力加权


class HG_Block(nn.Module):
    """HGNet 基础构建块
    关键结构：
        - 多层卷积堆叠
        - 特征聚合 (concat后接聚合卷积)
        - 残差连接(可选)
    参数说明：
    | in_chs | mid_chs | out_chs | layer_num | kernel_size | residual | light_block | agg |
    """
    def __init__(self, in_chs,  mid_chs,  out_chs, layer_num, kernel_size=3, 
                 residual=False, light_block=False, use_lab=False, agg='ese', drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        # 构建多层卷积
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # 特征聚合
        total_chs = in_chs + layer_num * mid_chs   # 计算总通道数，每个层的输出都会被保留并拼接
        if agg == 'se':    # SE 聚合分支
            # 压缩层：降维到 out_chs//2
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            # 激发层：恢复维度到 out_chs
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            # 聚合序列：压缩 → 激发
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:  # ESE 聚合分支（默认）​
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )

            att = EseModule(out_chs)   # 添加ESE注意力
            self.aggregation = nn.Sequential(  # 聚合序列：卷积 → 注意力
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity() # 在残差连接时随机丢弃路径，防止过拟合

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)    # 通道维度拼接
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity   # 带dropout的残差连接
        return x


class HG_Stage(nn.Module):
    """网络阶段模块
    功能：
        - 控制下采样
        - 包含多个HG_Block
    参数流程：
        in_chs -> [多个HG_Block] -> out_chs
    """
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(  # 下采样模块
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        # 构建多个HG_Block
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,   # 首个block不加残差
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    # 动态设置DropPath的丢弃概率，为元组或者列表时，用索引i获得，否则为单一值
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path, 
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



@register()
class HGNetv2(nn.Module):
    """HGNetv2 主干网络（多版本可配置）
    关键特性：
    - 支持B0-B6多种模型尺寸配置
    - 灵活的特征阶段返回机制
    - 预训练权重自动加载
    - 参数冻结控制(stem/全阶段)
    
    配置参数说明(arch_configs):
    |-- stem_channels: list[输入通道, 中间通道, 输出通道]
    |-- stage_config: dict{
        "stageN": [
            输入通道,        # int
            中间通道,        # int  
            输出通道,        # int
            块数量,          # 该阶段包含的HG_Block数量
            是否下采样,      # bool
            是否轻量块,      # bool
            卷积核尺寸,      # int
            每块层数         # 每个HG_Block的层数
        ]
    }
    |-- url: 预训练权重下载地址
    
    初始化参数说明：
    name: str          模型版本(B0-B6)
    use_lab: bool      是否使用可学习仿射变换
    return_idx: list   需要返回的特征阶段索引(从0开始)
    freeze_stem_only: bool 是否仅冻结stem参数
    freeze_at: int     冻结前N个阶段的参数(-1表示不冻结)
    freeze_norm: bool  是否冻结BN层参数
    pretrained: bool   是否加载预训练权重
    local_model_dir: str 本地模型存储路径
    """

    # 多版本模型架构配置字典
    arch_configs = {
        'B0': {    # 基础版配置
            'stem_channels': [3, 16, 16],    # Stem模块的[输入,中间,输出]通道
            'stage_config': {                # 四阶段配置
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],    # 格式解释见类注释
                "stage2": [64, 32, 256, 1, True, False, 3, 3],    # 下采样开启
                "stage3": [256, 64, 512, 2, True, True, 5, 3],    # 使用轻量块
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],  # 最终输出通道
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/'):
        super().__init__()
        # 参数初始化
        self.use_lab = use_lab         # 控制是否使用可学习仿射变换
        self.return_idx = return_idx   # 指定需要返回的特征阶段索引

        # 获取模型配置
        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        # 计算输出特征参数（用于下游任务）
        self._out_strides = [4, 8, 16, 32]      # 各阶段特征图下采样率
        self._out_channels = [stage_config[k][2] for k in stage_config]    # 各阶段输出通道数

        # 构建网络模块 -------------------------------------------------
        # 1. Stem模块（特征预处理）
        self.stem = StemBlock(
                in_chs=stem_channels[0],   # 输入通道数
                mid_chs=stem_channels[1],  # 中间通道数
                out_chs=stem_channels[2],  # 输出通道数
                use_lab=use_lab)           # 是否添加可学习仿射变换

        # 2. 多阶段特征提取
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,    # 该阶段包含的HG_Block数量
                    layer_num,    # 每个Block的层数
                    downsample,   # 是否下采样
                    light_block,  # 是否使用轻量块
                    kernel_size,  # 卷积核尺寸
                    use_lab))     # 可学习仿射变换
        # 参数冻结控制 -------------------------------------------------
        if freeze_at >= 0:                        # 冻结指定模块参数
            self._freeze_parameters(self.stem)    # 总是冻结stem
            if not freeze_stem_only:              # 扩展冻结范围
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:                # 冻结BN层参数
            self._freeze_norm(self)    # 递归遍历所有子模块

        # 预训练加载 -------------------------------------------------
        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                # 优先尝试本地加载
                model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    # If the file doesn't exist locally, download from the URL
                    # 分布式环境下处理（仅主进程下载）
                    if torch.distributed.get_rank() == 0:
                        print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        print(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                        # 同步所有进程
                        torch.distributed.barrier()
                    else:
                        torch.distributed.barrier()
                        state = torch.load(local_model_dir)

                    print(f"Loaded stage1 {name} HGNetV2 from URL.")
                # 加载权重到模型
                self.load_state_dict(state)

            except (Exception, KeyboardInterrupt) as e:
                # 异常处理（网络问题/文件损坏）
                if torch.distributed.get_rank() == 0:
                    print(f"{str(e)}")
                    logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                    logging.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()




    def _freeze_norm(self, m: nn.Module):
         """
        递归冻结所有BN层参数
        原理：将普通BN层替换为冻结参数的FrozenBatchNorm2d
        """
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
         """
        冻结模块所有可训练参数
        应用场景：冻结stem或早期stage的参数
        """
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        """前向传播流程
            输入：RGB图像张量 [B, 3, H, W]
            输出：指定阶段的特征图列表（默认返回stage1-3）
        """
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
