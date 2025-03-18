from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import typing as tp
import torch
from torch import nn

from quantization import QuantizedResult, BaseQuantizer
from vq import SplitResidualVectorQuantizer, ResidualVectorQuantizer
from resample import ConvDownsample1d, ConvTrUpsample1d
from streaming import StreamingModule, State, StateT
from compile import no_compile, CUDAGraphed


# 获取当前模块的日志记录器
logger = logging.getLogger()


class CompressionModel(StreamingModule[StateT]):
    """
    所有旨在作为音频分词器与语言模型一起使用的压缩模型的基类 API。

    该基类定义了压缩模型应实现的基本方法，包括前向传播、编码、解码等。
    所有继承自 `CompressionModel` 的子类必须实现这些抽象方法。

    Attributes:
        @abstractmethod forward (torch.Tensor) -> QuantizedResult:
            对输入张量进行前向传播，返回量化结果。

        @abstractmethod encode (torch.Tensor) -> torch.Tensor:
            对输入张量进行编码，参见 `MimiModel.encode`。

        @abstractmethod decode (torch.Tensor) -> torch.Tensor:
            对编码后的代码进行解码，参见 `MimiModel.decode`。

        @abstractmethod decode_latent (torch.Tensor) -> torch.Tensor:
            将离散代码解码到连续潜在空间。

        @property @abstractmethod channels (int):
            音频通道数。

        @property @abstractmethod frame_rate (float):
            帧率。

        @property @abstractmethod sample_rate (int):
            采样率。

        @property @abstractmethod cardinality (int):
            词汇表基数，即编码空间的大小。

        @property @abstractmethod num_codebooks (int):
            代码书数量。

        @property @abstractmethod total_codebooks (int):
            总代码书数量。

        @abstractmethod set_num_codebooks (int):
            设置量化器使用的活动代码书数量。
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedResult: 
        """
        对输入张量进行前向传播。

        Args:
            x (torch.Tensor): 输入音频张量。

        Returns:
            QuantizedResult: 量化结果，包含编码后的代码和其他相关信息。
        """
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入音频张量进行编码。

        参见 `MimiModel.encode` 方法。

        Args:
            x (torch.Tensor): 输入音频张量。

        Returns:
            torch.Tensor: 编码后的代码张量。
        """
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        对编码后的代码进行解码。

        参见 `MimiModel.decode` 方法。

        Args:
            codes (torch.Tensor): 编码后的代码张量。

        Returns:
            torch.Tensor: 解码后的音频张量。
        """
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """
        将离散代码解码到连续潜在空间。

        Args:
            codes (torch.Tensor): 离散代码张量。

        Returns:
            torch.Tensor: 解码后的连续潜在空间张量。
        """
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int): ...


@dataclass
class _MimiState(State):
    """
    Mimi模型的内部状态类。

    用于跟踪压缩模型的状态信息，包括CUDA图对象。

    Attributes:
        graphed_tr_enc (CUDAGraphed | None): 编码器的CUDA图对象。
        graphed_tr_dec (CUDAGraphed | None): 解码器的CUDA图对象。
    """
    graphed_tr_enc: CUDAGraphed | None
    graphed_tr_dec: CUDAGraphed | None


class MimiModel(CompressionModel[_MimiState]):
    """
    Mimi模型，对原始波形进行操作。

    该模型包含编码器、解码器、量化器等组件，支持流式处理和可选的Transformer模块。
    支持冻结编码器和解码器的权重，以及对量化器进行部分冻结。

    Args:
        encoder (nn.Module): 编码器网络。
        decoder (nn.Module): 解码器网络。
        quantizer (BaseQuantizer): 量化器网络。
        frame_rate (float): 量化表示的最终帧率。
        encoder_frame_rate (float): 编码器模型的帧率。
            注意，如果 `frame_rate != encoder_frame_rate`，则在量化和反量化前后，
            潜在空间将被线性重采样以匹配所需的 `frame_rate`。
        sample_rate (int): 音频采样率。
        channels (int): 音频通道数。
        causal (bool, optional): 是否使用因果版本的模型，默认为 `False`。
        encoder_transformer (nn.Module | None, optional): 编码器的可选Transformer，默认为 `None`。
        decoder_transformer (nn.Module | None, optional): 解码器的可选Transformer，默认为 `None`。
        resample_method (str, optional): 在量化器之前用于重采样潜在空间的方法，默认为 `"interpolate"`。
        upsample_channel_wise_bug (bool, optional): 控制上采样是否按通道进行。
            默认为 `True`，以复现原始实现中的错误。
        freeze_encoder (bool, optional): 是否冻结编码器权重，默认为 `False`。
        freeze_quantizer (bool, optional): 是否冻结量化器权重，默认为 `False`。
        freeze_quantizer_level (int, optional): 如果为正，则冻结量化器到该层级，默认为 `-1`（不冻结）。
        torch_compile_encoder_decoder (bool, optional): 如果为 `True`，则对编码器/解码器使用 `torch.compile`。
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: BaseQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: tp.Optional[nn.Module] = None,
        decoder_transformer: tp.Optional[nn.Module] = None,
        resample_method: str = "interpolate",
        upsample_channel_wise_bug: bool = True,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
        torch_compile_encoder_decoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder

        # 冻结编码器权重
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.encoder_transformer is not None:
                for p in self.encoder_transformer.parameters():
                    p.requires_grad = False
            for name, p in self.quantizer.named_parameters():
                if name.endswith("input_proj.weight"):
                    p.requires_grad = False
        
        # 冻结量化器权重
        if freeze_quantizer:
            self.quantizer.ema_frozen_(True)
        self.freeze_quantizer = freeze_quantizer
        self.freeze_quantizer_level = (
            freeze_quantizer_level
            if freeze_quantizer_level > 0
            else self.quantizer.num_codebooks
        )

        # 获取编码器的维度，用于重采样
        dimension = encoder.dimension

        # 检查重采样方法是否有效
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

        assert resample_method in [
            "interpolate",
            "conv",
            "avg_pool",
        ], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method

        # 如果编码器帧率与最终帧率不同，则进行重采样
        if encoder_frame_rate != frame_rate:
            assert not (
                causal and resample_method == "interpolate"
            ), "Cannot interpolate with causal model."
            if resample_method in ["conv", "avg_pool"]:
                assert (
                    self.encoder_frame_rate > self.frame_rate
                ), "Cannot upsample with conv."
                downsample_stride = self.encoder_frame_rate / self.frame_rate
                assert downsample_stride == int(
                    downsample_stride
                ), f"Only integer strides are supported, got {downsample_stride}"
                learnt = resample_method == "conv"
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                )
                if freeze_encoder:
                    for p in self.downsample.parameters():
                        p.requires_grad = False
                self.upsample = ConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                )

    def _init_streaming_state(self, batch_size: int) -> _MimiState:
        """
        初始化流式处理状态。

        Args:
            batch_size (int): 当前批次的样本数量。

        Returns:
            _MimiState: 初始化后的状态。
        """
        device = next(self.parameters()).device
        disable = device.type != 'cuda'
        graphed_tr_dec = None
        graphed_tr_enc = None
        if self.encoder_transformer is not None:
            # 创建编码器Transformer的CUDA图
            graphed_tr_enc = CUDAGraphed(self.encoder_transformer, disable=disable)
        if self.decoder_transformer is not None:
            # 创建解码器Transformer的CUDA图
            graphed_tr_dec = CUDAGraphed(self.decoder_transformer, disable=disable)
        return _MimiState(graphed_tr_enc, graphed_tr_dec)

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.cardinality

    def _to_framerate(self, x: torch.Tensor):
        """
        将编码器帧率转换为整体帧率。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 转换后的张量。
        """
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.downsample(x)

    def _to_encoder_framerate(self, x: torch.Tensor):
        """
        将整体帧率转换为编码器帧率。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 转换后的张量。
        """
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.upsample(x)

    @property
    def _context_for_encoder_decoder(self):
        """
        获取编码器/解码器的上下文管理器。

        Returns:
            contextlib.ContextManager: 上下文管理器。
        """
        if self.torch_compile_encoder_decoder:
            return nullcontext()
        else:
            return no_compile()

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        """
        前向传播方法。

        对输入音频张量进行编码、量化、反量化、解码，并返回量化结果。

        Args:
            x (torch.Tensor): 输入音频张量，形状为 `[B, C, T]`。

        Returns:
            QuantizedResult: 量化结果，包含编码后的代码、解码后的音频和其他指标。
        """
        assert x.dim() == 3
        length = x.shape[-1]
        extra_metrics: tp.Dict[str, torch.Tensor] = {}

        if self.freeze_quantizer:
            if isinstance(self.quantizer, SplitResidualVectorQuantizer):
                self.quantizer.rvq_first.eval()
                for i in range(
                    self.freeze_quantizer_level - self.quantizer.n_q_semantic
                ):
                    self.quantizer.rvq_rest.vq.layers[i].eval()
            elif isinstance(self.quantizer, ResidualVectorQuantizer):
                for i in range(self.freeze_quantizer_level):
                    self.quantizer.vq.layers[i].eval()
            else:
                raise ValueError(f"Unsupported quantizer type {type(self.quantizer)}")

        # 使用上下文管理器进行编码器/解码器的处理
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
        emb = self._to_framerate(emb)
        expected_length = self.frame_rate * length / self.sample_rate
        # 检查潜在空间的输出长度是否符合预期
        assert abs(emb.shape[-1] - expected_length) < 1, (
            emb.shape[-1],
            expected_length,
        )

        # 对编码后的嵌入进行量化
        q_res = self.quantizer(emb, self.frame_rate)
        emb = q_res.x
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)

        # 使用上下文管理器进行解码器的处理
        with self._context_for_encoder_decoder:
            out = self.decoder(emb)

        # 移除编码器和解码器添加的额外填充
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        # 将解码后的输出赋值回量化结果
        q_res.x = out
        q_res.metrics.update(extra_metrics)
        return q_res

    def _encode_to_unquantized_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        将一批波形投影到未量化的潜在空间。

        Args:
            x (torch.Tensor): 形状为 `[B, C, T]` 的浮点张量。

        Returns:
            未量化的嵌入。
        """
        assert (
            x.dim() == 3
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"
        state = self._streaming_state
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            if state is None:
                (emb,) = self.encoder_transformer(emb)
            else:
                assert state.graphed_tr_enc is not None
                (emb,) = state.graphed_tr_enc(emb)
        emb = self._to_framerate(emb)
        return emb

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        将给定的输入张量编码为量化表示。

        Args:
            x (torch.Tensor): 输入张量，形状为 `[B, C, T]`，其中：
                - B: 批次大小（Batch Size）
                - C: 通道数（Channels）
                - T: 时间步长度（Time Steps）

        Returns:
            codes (torch.Tensor): 编码后的代码张量，形状为 `[B, K, T]`，其中：
                - K: 使用的代码书数量（Number of Codebooks）
                - T: 时间步长度（Time Steps）
        """
        # 将输入张量投影到未量化的潜在空间
        emb = self._encode_to_unquantized_latent(x)
        # 使用量化器对嵌入进行编码
        codes = self.quantizer.encode(emb)
        # 返回编码后的代码
        return codes

    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """
        将一批波形投影到潜在空间。

        Args:
            x (torch.Tensor): 输入张量，形状为 `[B, C, T]`，其中：
                - B: 批次大小（Batch Size）
                - C: 通道数（Channels）
                - T: 时间步长度（Time Steps）
            quantize (bool, optional): 是否进行量化。默认为 `True`。

        Returns:
            torch.Tensor: 嵌入张量，可能是量化的，也可能不是。
        """
        # 将输入张量投影到未量化的潜在空间
        emb = self._encode_to_unquantized_latent(x)
        if not quantize:
            # 如果不进行量化，则返回未量化的嵌入
            return emb
        else:
            # 使用量化器对嵌入进行编码
            codes = self.quantizer.encode(emb)
            # 将编码后的代码解码回潜在空间
            return self.decode_latent(codes)

    def decode(self, codes: torch.Tensor):
        """
        将给定的代码解码为重构的表示。

        Args:
            codes (torch.Tensor): 输入代码张量，形状为 `[B, K, T]`，其中：
                - B: 批次大小（Batch Size）
                - K: 代码书数量（Number of Codebooks）
                - T: 时间步长度（Time Steps）

        Returns:
            out (torch.Tensor): 重构的音频张量，形状为 `[B, C, T]`，其中：
                - B: 批次大小（Batch Size）
                - C: 通道数（Channels）
                - T: 时间步长度（Time Steps）
        """
        # 获取当前流式处理状态
        state = self._streaming_state
        # 将代码解码到潜在空间
        emb = self.decode_latent(codes)
        # 将潜在空间的表示转换为编码器帧率
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            if state is None:
                # 如果没有流式状态，则直接应用解码器Transformer
                (emb,) = self.decoder_transformer(emb)
            else:
                assert state.graphed_tr_dec is not None
                # 如果有流式状态，则应用CUDA图加速的解码器Transformer
                (emb,) = state.graphed_tr_dec(emb)
        with self._context_for_encoder_decoder:
            # 使用解码器生成重构的音频
            out = self.decoder(emb)
        # out 包含编码器和解码器添加的额外填充
        return out

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """
        将离散代码解码到连续潜在空间。

        Args:
            codes (torch.Tensor): 离散代码张量。

        Returns:
            torch.Tensor: 解码后的连续潜在空间张量。
        """
        # 使用量化器将代码解码到潜在空间
        return self.quantizer.decode(codes)


class WrapperCompressionModel(CompressionModel[State]):
    """
    不依赖于外部框架的 CompressionModel 包装器的基类 API。

    该包装器类封装了一个现有的压缩模型，并提供了相同的方法接口，以便在不修改原始模型的情况下进行扩展或修改。

    Args:
        model (CompressionModel): 要包装的压缩模型。
    """

    def __init__(self, model: CompressionModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        """
        前向传播方法。

        直接调用被包装模型的 `forward` 方法。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            QuantizedResult: 量化结果。
        """
        return self.model.forward(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码方法。

        直接调用被包装模型的 `encode` 方法。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 编码后的代码。
        """
        return self.model.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        解码方法。

        直接调用被包装模型的 `decode` 方法。

        Args:
            codes (torch.Tensor): 输入代码。

        Returns:
            torch.Tensor: 解码后的音频。
        """
        return self.model.decode(codes)

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """
        将代码解码到潜在空间。

        直接调用被包装模型的 `decode_latent` 方法。

        Args:
            codes (torch.Tensor): 输入代码。

        Returns:
            torch.Tensor: 解码后的潜在空间表示。
        """
        return self.model.decode_latent(codes)

    def set_num_codebooks(self, n: int):
        """
        设置量化器使用的活动代码书数量。

        直接调用被包装模型的 `set_num_codebooks` 方法。

        Args:
            n (int): 活动代码书数量。
        """
        self.model.set_num_codebooks(n)

    @property
    def quantizer(self):
        return self.model.quantizer

    @property
    def channels(self) -> int:
        return self.model.channels

    @property
    def frame_rate(self) -> float:
        return self.model.frame_rate

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.cardinality

    @property
    def num_codebooks(self) -> int:
        return self.model.num_codebooks

    @property
    def total_codebooks(self) -> int:
        return self.model.total_codebooks
