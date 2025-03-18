from dataclasses import dataclass, field
import typing as tp
import torch
from torch import nn


@dataclass
class QuantizedResult:
    x: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class BaseQuantizer(nn.Module):
    """
    基础量化器类，所有量化器的基类。

    该类定义了量化器的基本接口，包括前向传播、编码、解码、属性访问等方法。
    所有继承自 `BaseQuantizer` 的子类必须实现这些抽象方法。

    Attributes:
        _ema_frozen (bool): 是否冻结指数移动平均（EMA）更新，默认为 `False`。
    """

    def __init__(self):
        super().__init__()
        self._ema_frozen = False

    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        """
        前向传播方法。

        给定输入张量 `x`，返回量化（或近似量化）表示，以及量化代码、带宽和任何损失惩罚项。
        最后，返回用于更新日志等的指标字典。
        必须传递帧率，以便正确计算带宽。

        Args:
            x (torch.Tensor): 输入张量。
            frame_rate (int): 帧率，用于计算带宽。

        Returns:
            QuantizedResult: 量化结果，包含量化后的张量、代码、带宽、惩罚项和指标。

        Raises:
            NotImplementedError: 如果子类未实现此方法。
        """
        raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用指定的采样率和带宽对给定的输入张量进行编码。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 编码后的代码。

        Raises:
            NotImplementedError: 如果子类未实现此方法。
        """
        raise NotImplementedError()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        将给定的代码解码为量化表示。

        Args:
            codes (torch.Tensor): 输入代码。

        Returns:
            torch.Tensor: 解码后的量化张量。

        Raises:
            NotImplementedError: 如果子类未实现此方法。
        """
        raise NotImplementedError()

    @property
    def cardinality(self) -> int:
        raise NotImplementedError()

    @property
    def total_codebooks(self) -> int:
        raise NotImplementedError()

    @property
    def num_codebooks(self) -> int:
        raise NotImplementedError()

    @property
    def semantic_quantizer(self) -> 'BaseQuantizer':
        """
        返回建模第一层（通常是语义层）的量化器。

        在这种情况下，它返回量化器本身。

        Returns:
            BaseQuantizer: 语义量化器。
        """
        return self

    @property
    def acoustic_quantizer(self) -> 'BaseQuantizer':
        """
        返回建模更高层（通常是声学层）的量化器。

        在这种情况下，它返回量化器本身。

        Returns:
            BaseQuantizer: 声学量化器。
        """
        return self

    def set_num_codebooks(self, n: int) -> None:
        raise NotImplementedError()

    @property
    def ema_frozen(self) -> bool:
        return self._ema_frozen

    def ema_frozen_(self, ema_frozen: bool) -> None:
        self._ema_frozen = ema_frozen


class DummyQuantizer(BaseQuantizer):
    """
    虚拟量化器（DummyQuantizer）。

    这是一个假的量化器，实际上并不执行任何量化操作。
    它将输入张量视为已量化表示，并返回与输入相同的代码。

    Args:
        dimension (int): 量化器的维度。
        input_dimension (int | None, optional): 输入的维度。如果未提供，则默认为 `dimension`。
        output_dimension (int | None, optional): 输出的维度。如果未提供，则默认为 `dimension`。
    """

    def __init__(
        self,
        dimension: int,
        input_dimension: tp.Optional[int] = None,
        output_dimension: tp.Optional[int] = None,
    ):
        super().__init__()
        # 设置量化器的维度
        self.dimension = dimension
        # 设置输入维度，默认为 `dimension`
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        # 输入投影层
        self.input_proj: torch.nn.Module
        # 输出投影层
        self.output_proj: torch.nn.Module

        # 设置输入投影层
        if self.input_dimension == self.dimension:
            # 如果维度相同，则使用恒等映射
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False  # 否则，使用1D卷积进行投影
            )

        # 设置输出投影层
        if self.input_dimension == self.dimension:
            # 如果维度相同，则使用恒等映射
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False  # 否则，使用1D卷积进行投影
            )

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        前向传播方法。

        在虚拟量化器中，实际不执行任何量化操作。
        将输入张量视为已量化表示，并返回与输入相同的代码。

        Args:
            x (torch.Tensor): 输入张量，形状为 `[B, C, T]`，其中：
                - B: 批次大小（Batch Size）
                - C: 通道数（Channels）
                - T: 时间步长度（Time Steps）
            frame_rate (int): 输入的帧率，用于计算带宽。

        Returns:
            QuantizedResult: 量化结果，其中：
                - `x`: 输入张量本身。
                - `codes`: 输入张量本身。
                - `bandwidth`: 计算得到的带宽。
        """
        # 在第二维增加一个维度，形状变为 `[B, 1, C, T]`
        q = x.unsqueeze(1)
        # 应用输入和输出投影
        x = self.output_proj(self.input_proj(x))
        # 计算带宽：每个元素32位，帧率乘以元素数量，再除以1000（转换为kbit/s）和批次大小
        return QuantizedResult(
            x, q, torch.tensor(q.numel() * 32 * frame_rate / 1000 / len(x)).to(x)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用指定的采样率和带宽对给定的输入张量进行编码。

        在虚拟量化器中，编码后的代码实际上与输入和量化表示相同，因为没有进行任何量化。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 编码后的代码，形状为 `[B, 1, C, T]`。
        """
        # 应用输入投影
        x = self.input_proj(x)
        # 在第二维增加一个维度，形状变为 `[B, 1, C, T]`
        return x.unsqueeze(1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        将给定的代码解码为量化表示。

        在虚拟量化器中，代码实际上与输入和量化表示相同，因为没有进行任何量化。

        Args:
            codes (torch.Tensor): 输入代码。

        Returns:
            torch.Tensor: 解码后的量化张量。
        """
        # 移除第二维，形状恢复为 `[B, C, T]`
        y = codes.squeeze(1)
        # 应用输出投影
        return self.output_proj(y)

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        return 1

    @property
    def num_codebooks(self):
        """Total number of codebooks."""
        return self.total_codebooks

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise AttributeError(
            "Cannot override the number of codebooks for the dummy quantizer"
        )

    @property
    def cardinality(self) -> int:
        """Cardinality of each codebook."""
        return 1
