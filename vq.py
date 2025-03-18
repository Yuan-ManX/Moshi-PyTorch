import math
import random
import typing as tp
import torch

from quantization import BaseQuantizer, QuantizedResult
from core_vq import ResidualVectorQuantization


class ResidualVectorQuantizer(BaseQuantizer):
    """
    残差向量量化器（Residual Vector Quantizer, RVQ）。

    RVQ 通过多个向量量化器（VQ）逐步量化输入信号，每个量化器处理前一个量化器的残差。
    这种方法可以提高量化的精度，并减少量化误差。

    Args:
        dimension (int, optional): 每个向量量化器的维度，默认为 `128`。
        input_dimension (int | None, optional): 输入的维度。如果未提供，则默认为 `dimension`。
        output_dimension (int | None, optional): 输出的维度。如果未提供，则默认为 `dimension`。
        n_q (int, optional): 使用的向量量化器数量，默认为 `8`。
        q_dropout (bool, optional): 在训练时是否随机丢弃量化器，默认为 `False`。
        no_quantization_rate (float, optional): 在训练时完全不应用量化的概率，默认为 `0.0`。
            即使应用了不量化，RVQ 的代码书仍然会接收到输入值以学习正确的代码书。
        bins (int, optional): 代码书的大小，即每个量化器的离散符号数量，默认为 `1024`。
        decay (float, optional): 代码书的指数移动平均衰减率，默认为 `0.99`。
        threshold_usage_ratio (float, optional): 定义替换质心的使用率阈值，默认为 `0.1`。
            该值表示在均匀分布下质心应获得的使用率的分数，因此它不依赖于批次大小等。
        replaced_usage_ratio (float, optional): 替换质心时使用的初始质心使用率，默认为 `1.0`。
            这用于避免质心被过快替换。
        codebook_offset (int, optional): 代码书索引的偏移量。这在如 `SplitResidualVectorQuantizer` 这样的多量化器情况下很有用，默认为 `0`。
        force_projection (bool, optional): 是否强制使用输入和输出投影，即使维度是常数，默认为 `False`。
        generator_seed (int | None, optional): 用于初始化不量化随机数生成器的种子，默认为 `None`。
    """

    def __init__(
        self,
        dimension: int = 128,
        input_dimension: tp.Optional[int] = None,
        output_dimension: tp.Optional[int] = None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        codebook_offset: int = 0,
        force_projection: bool = False,
    ):
        super().__init__()
        # 最大量化器数量
        self.max_n_q = n_q
        # 当前使用的量化器数量
        self.n_q = n_q
        # 是否启用量化器丢弃
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        # 量化器维度
        self.dimension = dimension
        # 输入维度
        self.input_dimension = input_dimension or dimension
        # 输出维度
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        # 指数移动平均衰减率
        self.decay = decay
        # 随机数生成器，用于量化器丢弃
        self.rng_dropout = random.Random(1234)
        # 输入投影层
        self.input_proj: torch.nn.Module
        # 输出投影层
        self.output_proj: torch.nn.Module

        # 设置输入投影层
        if self.input_dimension == self.dimension and not force_projection:
            # 如果维度相同且不强制投影，则使用恒等映射
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False  # 否则，使用1D卷积进行投影
            )

        # 设置输出投影层
        if self.output_dimension == self.dimension and not force_projection:
            # 如果维度相同且不强制投影，则使用恒等映射
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False  # 否则，使用1D卷积进行投影
            )

        # 初始化残差向量量化器
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            threshold_usage_ratio=threshold_usage_ratio,
            replaced_usage_ratio=replaced_usage_ratio,
            codebook_offset=codebook_offset,
        )

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        前向传播方法。

        对输入张量进行量化，并返回量化结果。

        Args:
            x (torch.Tensor): 输入张量，形状为 `[B, C, T]`，其中：
                - B: 批次大小（Batch Size）
                - C: 通道数（Channels）
                - T: 时间步长度（Time Steps）
            frame_rate (int): 输入的帧率（例如 `T = frame_rate * duration`），用于计算带宽。

        Returns:
            QuantizedResult: 量化结果，包含以下属性：
                - `x` (torch.Tensor): 量化后的张量，形状为 `[B, C, T]`。
                - `codes` (torch.Tensor): 量化代码，形状为 `[B, K, T]`，其中 `K` 是代码书数量。
                - `bw` (torch.Tensor): 量化张量的带宽，单位为每秒千比特（kbits per second）。
                - `penalty` (torch.Tensor): 承诺损失（commitment loss）。
                - `metrics` (dict): RVQ指标，特别是死代码替换的比率和熵。
        """
        # 获取当前量化器数量
        n_q = self.n_q
        # 应用输入投影
        x = self.input_proj(x)
        if self.training and self.q_dropout:
            # 在训练时随机选择量化器数量
            n_q = self.rng_dropout.randint(1, self.n_q)
        # 计算每个量化器的带宽
        bw_per_q = math.log2(self.bins) * frame_rate / 1000
        # 应用残差向量量化
        quantized, codes, commit_loss, metrics = self.vq(x, n_q=n_q)
        # 获取批次大小
        B, _, _ = quantized.shape
        if self.training and self.no_quantization_rate > 0:
            # 生成不量化掩码
            mask = (torch.rand(B, 1, 1, device=x.device) <= self.no_quantization_rate).float()
            # 应用不量化
            quantized = x * mask + (1 - mask) * quantized
        # 应用输出投影
        quantized = self.output_proj(quantized)
        # 转置代码张量以匹配 `[B, K, T]` 的形状
        codes = codes.transpose(0, 1)
        # codes 的形状为 `[B, K, T]`，其中 T 是帧数，K 是代码书数量。
        bw = torch.tensor(n_q * bw_per_q).to(x)
        # 计算总带宽
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss), metrics=metrics)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用指定的帧率和带宽对给定的输入张量进行编码。
        RVQ 编码方法设置适当的量化器数量，并返回每个量化器的索引。

        Args:
            x (torch.Tensor): 输入张量，形状为 `[B, C, T]`，其中：
                - B: 批次大小（Batch Size）
                - C: 通道数（Channels）
                - T: 时间步长度（Time Steps）

        Returns:
            torch.Tensor: 编码后的代码，形状为 `[B, K, T]`，其中：
                - K: 代码书数量（Number of Codebooks）
                - T: 时间步长度（Time Steps）
        """
        # 获取当前量化器数量
        n_q = self.n_q
        if x.shape[-1] == 0:
            # 如果输入长度为0，返回空的代码张量
            return torch.empty((x.shape[0], n_q, 0), device=x.device, dtype=torch.int64)
        # 应用输入投影
        x = self.input_proj(x)
        # 使用量化器对输入进行编码
        codes = self.vq.encode(x, n_q=n_q)
        # 转置代码张量以匹配 `[B, K, T]` 的形状
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        将给定的代码解码为量化表示。

        Args:
            codes (torch.Tensor): 输入代码，形状为 `[B, K, T]`，其中：
                - B: 批次大小（Batch Size）
                - K: 代码书数量（Number of Codebooks）
                - T: 时间步长度（Time Steps）

        Returns:
            torch.Tensor: 解码后的量化张量，形状为 `[B, C, T]`，其中：
                - C: 通道数（Channels）
                - T: 时间步长度（Time Steps）
        """
        # codes 的形状为 `[B, K, T]`，其中 T 是帧数，K 是代码书数量，vq.decode 期望的形状为 `[K, B, T]`。
        codes = codes.transpose(0, 1)  # 转置代码张量以匹配 `[K, B, T]` 的形状
        # 使用量化器对代码进行解码
        quantized = self.vq.decode(codes) 
        # 应用输出投影
        quantized = self.output_proj(quantized)
        return quantized

    @property
    def total_codebooks(self):
        return self.max_n_q

    @property
    def num_codebooks(self):
        return self.n_q

    def set_num_codebooks(self, n: int):
        assert n >= 0 and n <= self.max_n_q
        self.n_q = n

    @property
    def cardinality(self) -> int:
        return self.bins


class SplitResidualVectorQuantizer(BaseQuantizer):
    """
    分裂残差向量量化器（Split Residual Vector Quantizer）。

    该量化器将输入信号分为语义部分和声学部分，分别使用不同的残差向量量化器进行量化。
    这种方法允许对不同类型的信号成分进行更精细的量化处理。

    Args:
        n_q (int): 使用的残差向量量化器总数。
        n_semantic_q (int): 用于语义量化的残差向量量化器数量。
        **kwargs: 传递给 `ResidualVectorQuantizer` 的参数，这些参数在语义和声学量化器之间共享。
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        n_q_semantic: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, (
            f"Number of quantizers {n_q} must be larger "
            f"than the number of semantic quantizers {n_q_semantic}."
        )
        # 最大量化器数量
        self.max_n_q = n_q
        # 语义量化器数量
        self.n_q_semantic = n_q_semantic
        # 声学量化器数量
        self.n_q_acoustic = n_q - n_q_semantic
        # 获取量化器丢弃参数，默认值为False
        q_dropout = kwargs.pop("q_dropout", False)

        # 初始化语义量化器，强制使用投影层，不启用量化器丢弃
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )

        # 初始化声学量化器，强制使用投影层，根据需要启用量化器丢弃
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            codebook_offset=1,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )

    def _renorm_and_add(
        self,
        first_val: torch.Tensor,
        rest_val: torch.Tensor,
        n_q_semantic: int,
        n_q_acoustic: int,
    ):
        """
        对语义和声学量化器的值进行重归一化并相加。

        这允许纠正被量化器数量归一化的统计数据。为了归一化，我们使用实际使用的量化器数量，例如，考虑量化器丢弃。

        Args:
            first_val (torch.Tensor): 语义量化器的值。
            rest_val (torch.Tensor): 声学量化器的值。
            n_q_semantic (int): 语义量化器的数量。
            n_q_acoustic (int): 声学量化器的数量。

        Returns:
            torch.Tensor: 重归一化后的总和。
        """
        # 总量化器数量
        n_q = n_q_semantic + n_q_acoustic
        # 重归一化语义量化器的值
        renorm_first_val = first_val * n_q_semantic / n_q
        # 重归一化声学量化器的值
        renorm_rest_val = rest_val * n_q_acoustic / n_q
        # 返回归一化后的总和
        return renorm_first_val + renorm_rest_val

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        前向传播方法。

        对输入张量进行语义和声学量化，并返回完整的量化结果。

        Args:
            x (torch.Tensor): 输入张量，形状为 `[B, C, T]`，其中：
                - B: 批次大小（Batch Size）
                - C: 通道数（Channels）
                - T: 时间步长度（Time Steps）
            frame_rate (int): 输入的帧率（例如 `T = frame_rate * duration`），用于计算带宽。

        Returns:
            QuantizedResult: 完整的量化结果，包含以下属性：
                - `x` (torch.Tensor): 量化后的张量，形状为 `[B, C, T]`。
                - `codes` (torch.Tensor): 量化代码，形状为 `[B, K, T]`，其中 `K` 是代码书数量。
                - `bw` (torch.Tensor): 量化张量的带宽，单位为每秒千比特（kbits per second）。
                - `penalty` (torch.Tensor): 承诺损失（commitment loss）。
                - `metrics` (dict): RVQ指标，特别是死代码替换的比率和熵。
        """
        # 对输入进行语义量化
        semantic_result = self.rvq_first(x, frame_rate)
        if self.n_q == self.n_q_semantic:
            # 如果总量化器数量等于语义量化器数量，则返回语义量化结果
            return semantic_result
        # 对输入进行声学量化
        acoustic_result = self.rvq_rest(x, frame_rate)
        # 合并语义和声学量化结果
        full_quantized_emb = semantic_result.x + acoustic_result.x
        # 合并语义和声学量化代码
        full_quantized_codes = torch.cat(
            [semantic_result.codes, acoustic_result.codes], dim=1
        )
        # 这实际是使用的量化器数量，例如，考虑量化器丢弃。
        n_q_semantic = semantic_result.codes.shape[1]
        n_q_acoustic = acoustic_result.codes.shape[1]
        # 计算总带宽
        full_quantized_bandwidth = semantic_result.bandwidth + acoustic_result.bandwidth
        # 计算总承诺损失
        full_quantized_penalty = self._renorm_and_add(
            semantic_result.penalty, acoustic_result.penalty, n_q_semantic, n_q_acoustic
        )
        # 获取语义量化指标
        full_quantized_metrics = semantic_result.metrics
        # 合并语义和声学量化指标
        for key, value in acoustic_result.metrics.items():
            if key in full_quantized_metrics:
                full_quantized_metrics[key] = self._renorm_and_add(
                    full_quantized_metrics[key], value, n_q_semantic, n_q_acoustic
                )
            else:
                full_quantized_metrics[key] = value
        # 返回完整的量化结果
        return QuantizedResult(
            full_quantized_emb,
            full_quantized_codes,
            full_quantized_bandwidth,
            penalty=full_quantized_penalty,
            metrics=full_quantized_metrics,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量进行编码。

        对输入张量进行语义和声学量化，并合并代码。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 编码后的代码。
        """
        # 对输入进行语义量化编码
        codes = self.rvq_first.encode(x)
        if self.n_q > self.n_q_semantic:
            # 对输入进行声学量化编码
            acoustic_codes = self.rvq_rest.encode(x)
            # 合并语义和声学量化代码
            codes = torch.cat([codes, acoustic_codes], dim=1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        对编码后的代码进行解码。

        对编码后的代码进行语义和声学量化解码，并合并结果。

        Args:
            codes (torch.Tensor): 编码后的代码。

        Returns:
            torch.Tensor: 解码后的量化表示。
        """
        # codes 是 [B, K, T]，其中 T 是帧数，K 是代码书数量。
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        # 对语义量化代码进行解码
        if codes.shape[1] > self.n_q_semantic:
            # 对声学量化代码进行解码并合并
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized

    @property
    def total_codebooks(self):
        return self.rvq_first.max_n_q + self.rvq_rest.max_n_q

    @property
    def num_codebooks(self):
        return self.rvq_first.num_codebooks + self.rvq_rest.num_codebooks

    @property
    def n_q(self):
        return self.rvq_first.n_q + self.rvq_rest.n_q

    @property
    def dimension(self):
        return self.rvq_first.dimension

    @property
    def semantic_quantizer(self) -> ResidualVectorQuantizer:
        return self.rvq_first

    @property
    def acoustic_quantizer(self) -> ResidualVectorQuantizer:
        return self.rvq_rest

    def set_num_codebooks(self, n: int):
        assert n >= self.n_q_semantic and n <= self.total_codebooks
        self.rvq_rest.set_num_codebooks(n - self.n_q_semantic)

    @property
    def cardinality(self) -> int:
        assert self.rvq_rest.cardinality == self.rvq_first.cardinality
        return self.rvq_first.cardinality
