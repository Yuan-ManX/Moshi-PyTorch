from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
import logging
import typing as tp
import torch
from torch import nn

from base import ConditionProvider, ConditionFuser, ConditionTensors
from sampling import sample_token
from compile import CUDAGraphed
import quantize
from streaming import StreamingContainer, StreamingModule, State
from transformer import (
    StreamingTransformer,
    quantize_transformer,
    create_norm_fn,
)


# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class ScaledEmbedding(nn.Embedding):
    """
    对嵌入层进行缩放以提升学习率（通过 `scale` 参数）。

    该类继承自 `nn.Embedding`，并添加了以下功能：
    - 可选的层归一化（Layer Normalization）
    - 特殊索引用于输出零
    - 低秩嵌入以减少权重数量（适用于非常大的词汇表）

    Args:
        num_embeddings (int): 嵌入字典的大小，即词汇表的大小。
        embedding_dim (int): 每个嵌入向量的维度。
        norm (bool, optional): 如果为 True，则在嵌入层之后应用层归一化。默认为 False。
        zero_idx (int, optional): 特殊索引，表示对应的输出应严格为零。默认为 -1。
        low_rank (int | None, optional): 如果提供，则使用低秩嵌入，通过一个线性层将嵌入维度
                                         转换为目标维度。这对于减少非常大的词汇表的权重数量非常有效。
                                         默认为 None。
        *args: 传递给 `nn.Embedding` 的其他位置参数。
        **kwargs: 传递给 `nn.Embedding` 的其他关键字参数。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 *args, norm: bool = False, zero_idx: int = -1,
                 low_rank: int | None = None, **kwargs):
        super().__init__(num_embeddings, low_rank or embedding_dim, *args, **kwargs)
        # 初始化归一化层
        self.norm = None
        if norm:
            # 使用自定义的归一化函数创建层归一化层
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        # 确保 zero_idx 为负值
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        # 特殊索引，用于输出零
        self.zero_idx = zero_idx
        # 初始化低秩线性层
        self.low_rank = None
        if low_rank is not None:
            # 创建一个线性层，将 low_rank 维度的输入转换为 embedding_dim 维度的输出
            self.low_rank = nn.Linear(low_rank, embedding_dim, bias=False)

    def forward(self, input, *args, **kwargs):
        """
        前向传播方法。

        Args:
            input (torch.Tensor): 输入张量，形状为 `[*,]`，其中 `*` 表示任意数量的维度。
            *args: 其他位置参数，传递给父类的 `forward` 方法。
            **kwargs: 其他关键字参数，传递给父类的 `forward` 方法。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同，但嵌入维度为 `embedding_dim`。
        """
        # 判断输入张量中哪些位置等于 zero_idx
        is_zero = input == self.zero_idx
        # 创建一个全零张量，形状为 `[1,]`，用于替换零索引的位置
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        # 将输入张量中小于零的值裁剪到零，确保索引为非负
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        # 如果启用了归一化，则对嵌入结果应用层归一化
        if self.norm is not None:
            y = self.norm(y)
        # 将零索引的位置替换为全零张量
        y = torch.where(is_zero[..., None], zero, y)
        # 如果启用了低秩嵌入，则应用线性层进行转换
        if self.low_rank is not None:
            y = quantize.linear(self.low_rank, y)
        return y


class LMModel(StreamingContainer):
    """
    基于Transformer的用于多代码流（多模态）的语言模型。

    该模型能够处理多个并行输入流，并在Depformer（一种依赖关系Transformer）中进行建模。
    支持文本和音频等多种模态的输入，并能够处理大规模词汇表。

    Args:
        delays (List[int], optional): 每个代码书的延迟列表，默认为 `[0]`。
        n_q (int, optional): 并行输入流的数量，默认为 `8`。
        dep_q (int, optional): 在Depformer中并行建模的流数量，默认为 `8`。
        card (int, optional): 词汇表基数（词汇表大小），默认为 `1024`。
        text_card (int, optional): 文本词汇表的基数，默认为 `32000`。
        dim (int, optional): Transformer编码器的维度，默认为 `128`。
        num_heads (int, optional): Transformer编码器的头数，默认为 `8`。
        hidden_scale (int, optional): Transformer编码器中隐藏前馈层的缩放因子，默认为 `4`。
        norm (str, optional): 归一化方法，默认为 `"layer_norm"`。
        norm_emb (bool, optional): 是否对嵌入进行归一化，默认为 `False`。
        bias_proj (bool, optional): 输出投影是否使用偏置，默认为 `False`。
        depformer_dim (int, optional): Depformer的维度，默认为 `256`。
        depformer_dim_feedforward (int | list[int] | None, optional): 
            Depformer中前馈层的维度。如果为 `None`，则默认为 `hidden_scale * depformer_dim`。
            默认为 `None`。
        depformer_multi_linear (bool, optional): 
            如果为 `True`，则对每个代码书使用一个线性层，将主Transformer的输出投影到Depformer的潜在空间。
            默认为 `False`。
        depformer_weights_per_step (bool, optional): 
            是否为每个步骤分配不同的权重，默认为 `False`。
        depformer_weights_per_step_schedule (list[int] | None, optional): 
            映射 `CODEBOOK_INDEX -> WEIGHT_INDEX`，允许为不同的代码书分配不同的权重。
            默认为 `None`。
        depformer_low_rank_embeddings (int | None, optional): 
            如果提供，则使用低秩嵌入，通过一个线性层进行投影。
            默认为 `None`。
        depformer_pos_emb (str, optional): 
            Depformer的位置嵌入方法，默认为 `"sin"`。
        existing_text_padding_id (int | None, optional): 
            如果提供，则使用不同的标记作为初始文本标记和文本填充标记。
            默认为 `None`。
        context (int | None, optional): 
            上下文长度，默认为 `None`。
        condition_provider (ConditionProvider | None, optional): 
            条件提供者，默认为 `None`。
        fuser (ConditionFuser | None, optional): 
            条件融合器，默认为 `None`。
        quantize (bool, optional): 
            是否对模型进行量化，默认为 `False`。
        device (torch.device | None, optional): 
            计算设备，默认为 `None`。
        dtype (torch.dtype | None, optional): 
            数据类型，默认为 `None`。
        **kwargs: 
            其他传递给Transformer编码器的关键字参数。
    """

    def __init__(
        self,
        delays: tp.List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_weights_per_step_schedule: list[int] | None = None,
        depformer_low_rank_embeddings: int | None = None,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: tp.Optional[int] = None,
        context: tp.Optional[int] = None,
        condition_provider: tp.Optional[ConditionProvider] = None,
        fuser: tp.Optional[ConditionFuser] = None,
        quantize: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()

        # 初始化参数
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks, "unexpected number of delays"
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule

        # 检查延迟列表长度是否与代码书数量一致
        if depformer_weights_per_step_schedule is not None:
            assert len(depformer_weights_per_step_schedule) == dep_q
        kwargs["context"] = context

        # 创建嵌入层工厂
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )

        # 创建音频嵌入层列表
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )
        
        # 计算文本嵌入层的额外标记数量
        extra_text = self.existing_text_padding_id is None
        
        # 创建文本嵌入层
        self.text_emb = EmbeddingFactory(text_card + 1, dim)

        # 创建文本线性层，用于将嵌入维度转换为文本词汇表大小
        self.text_linear = nn.Linear(dim, text_card + extra_text, bias=bias_proj)
        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }

        # 创建主Transformer模型
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            quantize=quantize,
            **main_kwargs,
        )

        # 创建输出归一化层
        self.out_norm = create_norm_fn(norm, dim)
        # 标记是否使用多线性层
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # 对每个codebook使用一个线性层
            num_in = dep_q
            if depformer_weights_per_step_schedule:
                num_in = max(depformer_weights_per_step_schedule) + 1
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(num_in)]
            )
        else:
            # 使用单个线性层
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )

        # 创建Depformer嵌入层工厂
        EmbeddingFactory = partial(EmbeddingFactory, low_rank=depformer_low_rank_embeddings)
        
        # 创建Depformer音频嵌入层列表
        self.depformer_emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )

        # 创建Depformer文本嵌入层
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)

        # 设置Depformer前馈层维度
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)

        # 创建Depformer模型
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            weights_per_step_schedule=depformer_weights_per_step_schedule,
            quantize=quantize,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        
        # 设置Depformer的流式处理为独立模式
        self.depformer.set_streaming_detached(True)
        # 更新维度为Depformer的维度
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        # 创建Depformer输出线性层列表
        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )

        # 初始化条件提供者和融合器
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.to(device=device, dtype=dtype)

        # 如果启用了量化，则对Transformer模型进行量化
        if quantize:
            quantize_transformer(self)

    @property
    def initial_token_id(self) -> int:
        """
        序列开始的标记ID（音频）。
        """
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """
        序列开始的标记ID（文本）。
        """
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """
        文本填充的标记ID。
        """
        if self.existing_text_padding_id is None:
            return self.text_card
        else:
            return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """
        可选地标记单词的最后一个填充步骤的标记ID。
        """
        return 0

    @property
    def zero_token_id(self) -> int:
        """
        输入标记中的特殊值，表示不应对该值进行采样，并且不应向模型提供输入。
        """
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """
        在提示中提供的特殊值，表示应预测并采样该特定值。这允许部分教师强制，通过生成一种模态，固定另一种模态。
        """
        return -2

    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        """
        返回将提供给模型以预测第一个时间步的初始标记。
        输出形状为 `[B, K, 1]`。

        Returns:
            torch.Tensor: 初始标记张量。
        """
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token

    def forward_text(
        self,
        sequence: torch.Tensor, sum_condition: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        处理文本输入的前向传播方法。

        Args:
            sequence (torch.Tensor): 输入序列，形状为 `[B, K, S]`，其中：
                - B: 批次大小
                - K: codebook数量
                - S: 序列长度
            sum_condition (torch.Tensor | None, optional): 条件张量，默认为 `None`。

        Returns:
            tuple: Transformer输出和文本逻辑输出。
        """
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(input_sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb
        if sum_condition is not None:
            input_ = input_ + sum_condition.to(input_)
        transformer_out = self.transformer(input_)

        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = quantize.linear(self.text_linear, transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        处理Depformer输入的前向传播方法。

        Args:
            depformer_cb_index (int): Depformer的代码书索引。
            sequence (torch.Tensor): 输入序列，形状为 `[B, K, S]`，其中：
                - B: 批次大小
                - K: codebook数量
                - S: 序列长度
            transformer_out (torch.Tensor): Transformer输出。

        Returns:
            torch.Tensor: Depformer逻辑输出。
        """
        B, K, S = sequence.shape
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: tp.Optional[torch.Tensor] = None
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            in_index = depformer_cb_index
            if self.depformer_weights_per_step_schedule is not None:
                in_index = self.depformer_weights_per_step_schedule[in_index]
            depformer_input = quantize.linear(self.depformer_in[in_index], depformer_input)
        else:
            depformer_input = quantize.linear(self.depformer_in[0], depformer_input)
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        assert last_token_input is not None
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        dep_output = self.depformer(depformer_input)
        logits = quantize.linear(self.linears[depformer_cb_index], dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits


@dataclass
class _LMGenState(State):
    """
    语言模型生成器的内部状态类。

    用于跟踪生成过程中的状态信息，包括批次大小、缓存、初始标记、CUDA图、条件总和、偏移量等。

    Attributes:
        batch_size (int): 当前批次的样本数量。
        cache (torch.Tensor): 缓存张量，用于存储生成的标记，形状为 `[batch_size, num_codebooks, max_delay + 2]`。
        initial (torch.Tensor): 初始标记张量，形状为 `[batch_size, num_codebooks, 1]`。
        graphed_main (CUDAGraphed): 主Transformer的前向传播CUDA图。
        graphed_depth (CUDAGraphed): Depformer前向传播的CUDA图。
        condition_sum (torch.Tensor | None, optional): 条件张量的总和，默认为 `None`。
        offset (int, optional): 当前生成步骤的偏移量，默认为 `0`。
        exit_stack (ExitStack): 上下文管理器堆栈，用于管理多个上下文。
        reset_callback (Callable[[], None] | None, optional): 重置回调函数，默认为 `None`。
    """
    batch_size: int
    cache: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_depth: CUDAGraphed
    condition_sum: torch.Tensor | None = None
    offset: int = 0
    exit_stack: ExitStack = field(default_factory=ExitStack)
    reset_callback: tp.Callable[[], None] | None = None

    def reset(self):
        """
        重置生成状态。

        将偏移量重置为 `0`，并调用重置回调函数（如果存在）。
        """
        self.offset = 0
        if self.reset_callback is not None:
            self.reset_callback()

    def __enter__(self):
        """
        进入上下文管理器。

        进入 `exit_stack` 的上下文。
        """
        self.exit_stack.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        """
        退出上下文管理器。

        退出 `exit_stack` 的上下文，并处理异常。
        """
        self.exit_stack.__exit__(exc_type, exc_value, traceback)


class LMGen(StreamingModule[_LMGenState]):
    """
    基于Transformer的语言模型生成器。

    该生成器使用 `LMModel` 进行文本和音频的生成，支持流式处理和条件生成。

    Args:
        lm_model (LMModel): 使用的语言模型。
        use_sampling (bool, optional): 是否使用采样进行生成，默认为 `True`。
        temp (float, optional): 音频生成温度，默认为 `0.8`。
        temp_text (float, optional): 文本生成温度，默认为 `0.7`。
        top_k (int, optional): 音频生成中使用的top-k值，默认为 `250`。
        top_k_text (int, optional): 文本生成中使用的top-k值，默认为 `25`。
        cfg_coef (float, optional): 分类器自由引导（CFG）系数，默认为 `1.0`。
        check (bool, optional): 是否进行生成检查，默认为 `False`。
        condition_tensors (ConditionTensors | None, optional): 条件张量，默认为 `None`。
    """
    def __init__(
        self,
        lm_model: LMModel,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        cfg_coef: float = 1.,
        check: bool = False,
        condition_tensors: ConditionTensors | None = None,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.lm_model.set_streaming_detached(True)
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.cfg_coef = cfg_coef
        self.check = check
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )
        self.condition_tensors = condition_tensors
        if self.cfg_coef != 1.:
            assert self.lm_model.fuser is not None, "Model has no fuser, cannot do CFG."
            assert self.condition_tensors, "Missing condition tensors for CFG."

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        """
        初始化流式处理状态。

        Args:
            batch_size (int): 当前批次的样本数量。

        Returns:
            _LMGenState: 初始化后的生成状态。
        """
        lm_model = self.lm_model
        # 获取初始标记
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )

        if self.lm_model.fuser is None:
            assert not self.condition_tensors
            condition_sum = None
        else:
            assert self.condition_tensors is not None
            # 计算条件总和
            condition_sum = self.lm_model.fuser.get_sum(self.condition_tensors)

        disable = lm_model.device.type != 'cuda'
        # 创建主Transformer的前向传播CUDA图
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        state = _LMGenState(
            batch_size, cache, initial, graphed_main, graphed_depth,
            # 创建生成状态
            condition_sum=condition_sum)

        if self.cfg_coef != 1.:
            batch_size *= 2
            if state.condition_sum is not None:
                assert state.condition_sum.shape[0] == batch_size, "CFG requires 2x more conditions."
        # 进入流式处理的上下文
        state.exit_stack.enter_context(self.lm_model.streaming(batch_size))
        state.reset_callback = self.lm_model.reset_streaming
        return state

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor) -> torch.Tensor | None:
        """
        执行生成器的一步前向传播。

        该方法接收用户输入的标记，更新缓存，并生成下一个标记。

        Args:
            input_tokens (torch.Tensor): 用户输入的标记，形状为 `[B, K, T]`，其中：
                - B: 批次大小（Batch Size）
                - K: 输入流的数量（Number of Input Streams）
                - T: 时间步长度（Time Steps），应为 1

        Returns:
            torch.Tensor | None: 生成的标记张量，形状为 `[B, K, T]`，如果当前步骤尚未生成完整输出，则返回 `None`。
        """
        # 获取当前流式处理状态
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        # 获取语言模型
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        # 获取批次大小、输入流数量和时间步长度
        B, Ki, S = input_tokens.shape
        assert B == state.batch_size, f"Got a batch size {B}, expected {state.batch_size}"
        assert S == 1, "Only support being given steps one by one."
        # 计算需要的输入流数量
        needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1
        assert (
            Ki == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

        # 获取缓存张量的时间步维度大小
        CT = state.cache.shape[2]

        for q_other in range(input_tokens.shape[1]):
            # 计算当前输入流对应的代码书索引
            k = lm_model.dep_q + 1 + q_other
            # 获取对应的延迟
            delay = lm_model.delays[k]
            # 计算写入位置，考虑延迟
            write_position = (state.offset + delay) % CT
            # 将输入标记写入缓存
            state.cache[:, k, write_position : write_position + 1] = input_tokens[
                :, q_other
            ]

        # 计算当前时间步的缓存位置
        position = state.offset % CT
        for k, delay in enumerate(lm_model.delays):
            # 仅在生成器的最开始几步，为延迟的声学标记填充初始标记
            if state.offset <= delay:
                state.cache[:, k, position] = state.initial[:, k, 0]
        # 从缓存中提取当前时间步的输入
        input_ = state.cache[:, :, position : position + 1]

        if self.check:
            # 检查是否在生成过程中引入了未生成的标记
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        if self.cfg_coef != 1.:
            # 如果使用CFG，则重复输入以进行双重处理
            input_ = input_.repeat(2, 1, 1)
        # 通过主Transformer处理输入
        transformer_out, text_logits = state.graphed_main(input_, state.condition_sum)
        if self.cfg_coef != 1.:
            # 分割逻辑输出
            logits, logits_null = text_logits.chunk(2)
            # 应用CFG系数
            text_logits = logits_null + (logits - logits_null) * self.cfg_coef
        # text_logits 的形状应为 `[B, K_text=1, T=1, Card_text]`
        text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert text_token.dim() == 3, text_token.shape
        assert text_token.shape[2] == 1
        assert text_token.shape[1] == 1, "Only one text stream supported."
        # 获取形状为 `[B]` 的文本标记
        text_token = text_token[:, 0, 0]  # shape is [B]
        # 通过Depformer生成音频标记
        audio_tokens = state.graphed_depth(text_token, transformer_out)

        # 确保不覆盖提示标记，只覆盖未生成的标记
        state.offset += 1  # 增加偏移量
        position = state.offset % CT  # 计算新的缓存位置
        state.cache[:, 0, position] = text_token  # 将文本标记写入缓存
        state.cache[:, 1 : lm_model.dep_q + 1, position] = audio_tokens  # 将音频标记写入缓存

        if state.offset <= self.max_delay:
            # 如果偏移量小于最大延迟，则返回 `None`，表示尚未生成完整输出
            return None
        B = state.cache.shape[0]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        # 计算要提取的缓存索引
        index = (
            ((state.offset - self.max_delay + gen_delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        # 从缓存中提取生成的标记
        out = state.cache.gather(dim=2, index=index)
        # 返回生成的标记
        return out

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行Depformer的前向传播步骤。

        该方法接收文本标记和Transformer的输出，生成Depformer的输出。

        Args:
            text_token (torch.Tensor): 文本标记，形状为 `[B,]`。
            transformer_out (torch.Tensor): Transformer的输出。

        Returns:
            torch.Tensor: Depformer的输出，形状为 `[B, dep_q]`。
        """
        # 获取批次大小
        B, = text_token.shape
        B_cfg = B
        if self.cfg_coef != 1.:
            # 如果使用CFG，则批次大小乘以2
            B_cfg = 2 * B
        # 初始化前一个标记为当前文本标记
        prev_token = text_token
        lm_model = self.lm_model
        # 初始化Depformer生成的标记列表
        depformer_tokens: list[torch.Tensor] = []
        assert not lm_model.depformer.is_streaming

        # 进入Depformer的流式处理上下文
        with lm_model.depformer.streaming(B_cfg):
            assert lm_model.depformer.is_streaming
            for cb_index in range(lm_model.dep_q):
                # 准备输入，形状为 `[B, 1, 1]`
                input_ = prev_token[:, None, None]
                if self.cfg_coef != 1.:
                    # 如果使用CFG，则重复输入
                    input_ = input_.repeat(2, 1, 1)
                # 通过Depformer生成逻辑输出
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                if self.cfg_coef != 1.:
                    # 分割逻辑输出
                    logits, logits_null = logits.chunk(2)
                    # 应用CFG系数
                    logits = logits_null + (logits - logits_null) * self.cfg_coef
                # 根据逻辑输出采样下一个标记
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                # 获取形状为 `[B]` 的下一个标记
                next_token = next_token[:, 0, 0]  # shape is B
                # 将下一个标记添加到列表中
                depformer_tokens.append(next_token)
                # 更新前一个标记
                prev_token = next_token

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        # 将标记堆叠成形状为 `[B, dep_q]` 的张量
        out = torch.stack(depformer_tokens, dim=1)
        assert out.shape == (B, lm_model.dep_q), out.shape
        return out
