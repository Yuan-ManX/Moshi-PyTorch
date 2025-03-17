import math
import torch
from torch import nn

from compile import torch_compile_lazy


@torch_compile_lazy
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    max_period: float = 10_000,
    time_before_heads: bool = False,
):
    """
    对查询和键张量应用旋转位置编码（RoPE）。

    Args:
        q (torch.Tensor): 查询张量，形状为 `[B, T, H, D]` 或 `[B, H, T, D]`。
        k (torch.Tensor): 键张量，形状与 `q` 相同。
        offset (torch.Tensor): 当前偏移量，用于计算位置编码，形状为标量或可广播到 `[T]`。
        max_period (float, optional): 旋转频率的最大周期，默认为 10,000。
        time_before_heads (bool, optional): 指示输入张量的维度顺序。默认为 False。

    Returns:
        tuple: 旋转后的查询和键张量，形状与输入相同。
    """
    # 判断输入张量的维度顺序
    if time_before_heads:
        B, T, H, D = q.shape
    else:
        B, H, T, D = q.shape

    # 确保张量形状和维度符合要求
    assert k.shape == q.shape
    assert D > 0
    assert D % 2 == 0
    assert max_period > 0

    # 计算旋转频率
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32) # 生成 [0, 1, 2, ..., D/2 -1]
    # 计算频率，形状为 [D/2]
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D)) 
    # 生成位置索引张量
    ts = offset.float() + torch.arange(T, device=q.device, dtype=torch.float32) # 生成 [0, 1, 2, ..., T-1] + offset
    if time_before_heads:
        # 调整形状为 [T, 1, 1]
        ts = ts.view(-1, 1, 1)
    else:
        # 调整形状为 [1, T, 1]
        ts = ts.view(1, -1, 1)

    # 调整查询和键张量的形状以分离实部和虚部
    # 获取除最后一个维度外的所有维度
    dims = q.shape[:-1]
    # 形状变为 [B, T, H, D/2, 2] 或 [B, H, T, D/2, 2]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)

    # 分离实部和虚部
    # 查询的实部，形状为 [B, T, H, D/2] 或 [B, H, T, D/2]
    qr = q[..., 0].float()
    qi = q[..., 1].float()

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    # 计算旋转矩阵的余弦和正弦部分
    # 余弦部分，形状为 [T, 1, 1] 或 [1, T, 1]
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)

    # 应用旋转矩阵
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    # 确保输出张量的数据类型与输入相同
    dtype = q.dtype
    # 合并实部和虚部，形状为 [B, T, H, D/2, 2] 或 [B, H, T, D/2, 2]
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    # 返回旋转后的查询和键张量，形状与输入相同
    return qo.view(*dims, D), ko.view(*dims, D)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）模块。

    参考文献:
        Su et al., 2022

    Args:
        max_period (float, optional): 旋转频率的最大周期，默认为 10,000。
    """
    def __init__(self, max_period: float = 10000.0):
        super().__init__()
        # 初始化最大周期
        self.max_period = max_period

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
        time_before_heads: bool = False,
    ):
        """
        对查询和键张量应用旋转位置编码（RoPE）。

        Args:
            q (torch.Tensor): 查询张量，形状为 `[B, T, H, D]` 或 `[B, H, T, D]`。
            k (torch.Tensor): 键张量，形状与 `q` 相同。
            offset (torch.Tensor): 当前偏移量，用于计算位置编码，形状为标量或可广播到 `[T]`。
            time_before_heads (bool, optional): 指示输入张量的维度顺序。默认为 False。

        Returns:
            tuple: 旋转后的查询和键张量，形状与输入相同。
        """
        # 调用 `apply_rope` 函数进行旋转编码
        return apply_rope(q, k, offset, self.max_period, time_before_heads)
