import torch


def multinomial(
    input: torch.Tensor, num_samples: int, replacement=False, *, generator=None
):
    """
    对输入张量进行多类别多项式采样，支持任意维度和最后一个维度的候选数量。

    该函数对输入张量 `input` 的最后一个维度进行多类别多项式采样，生成 `num_samples` 个样本。
    如果 `replacement` 为 `False`，则进行无放回抽样。

    Args:
        input (torch.Tensor): 包含概率分布的输入张量，形状为 `[..., C]`，其中 `C` 是候选数。
        num_samples (int): 要采样的样本数量。
        replacement (bool, optional): 是否进行有放回抽样。默认为 `False`（无放回）。
        generator (torch.Generator, optional): 用于采样的伪随机数生成器。默认为 `None`，使用默认随机数生成器。

    Returns:
        torch.Tensor: 采样结果，张量形状为 `[..., num_samples]`，其中最后一个维度包含 `num_samples` 个索引，
            这些索引是从输入张量的最后一个维度的多类别多项式概率分布中采样的。
    """
    # 将输入张量展平为二维张量，形状为 `[N, C]`
    input_ = input.reshape(-1, input.shape[-1])

    if replacement or num_samples != 1:
        # 如果进行有放回抽样，或者样本数量不为1，则使用 `torch.multinomial` 进行采样
        output_ = torch.multinomial(
            input_,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
    else:
        # 否则，进行无放回且样本数量为1的采样
        q = torch.empty_like(input_).exponential_(1, generator=generator)
        # 计算 `input_` 除以指数分布张量
        q = input_ / q
        output_ = q.argmax(dim=-1, keepdim=True)
    # 将输出张量重塑为与输入张量相同的维度，除了最后一个维度被替换为 `num_samples`
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """
    从输入概率张量的最后一个维度的前 `k` 个值中采样下一个标记。

    该函数首先在输入张量的最后一个维度上找到前 `k` 个最大值，然后对这些值进行多项式采样。

    Args:
        probs (torch.Tensor): 输入概率张量，候选标记位于最后一个维度。
        k (int): “top-k” 中的 `k`，即选择前 `k` 个概率最高的候选。

    Returns:
        torch.Tensor: 采样的标记。
    """
    # 在最后一个维度上找到前 `k` 个最大概率及其索引
    probs, indices = torch.topk(probs, k, dim=-1)
    # 对前 `k` 个概率进行多项式采样，采样1个样本
    next_token = multinomial(probs, num_samples=1)
    # 根据采样的索引从 `indices` 中获取实际的标记
    next_token = indices.gather(-1, next_token)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    从输入概率张量的最后一个维度的前 `p` 概率中采样下一个标记。

    该函数实现了 top-p（核采样）采样方法。首先对输入概率进行排序，然后累加概率，找到累积概率首次超过 `p` 的位置。
    然后，将概率低于该位置的候选标记的概率设为0，并重新归一化剩余概率。最后，对归一化后的概率进行多项式采样。

    Args:
        probs (torch.Tensor): 输入概率张量，候选标记位于最后一个维度。
        p (float): top-p 中的 `p`，表示累积概率的阈值。

    Returns:
        torch.Tensor: 采样的标记。
    """
    # 对输入概率进行降序排序
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算累积概率
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 创建一个掩码，标记累积概率超过 `p` 的位置
    mask = probs_sum - probs_sort > p
    # 将超过 `p` 的概率设为0
    probs_sort *= (~mask).float()
    # 重新归一化剩余概率
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 对归一化后的概率进行多项式采样，采样1个样本
    next_token = multinomial(probs_sort, num_samples=1)
    # 根据采样的索引从 `probs_idx` 中获取实际的标记
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_token(
    logits: torch.Tensor,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """
    给定形状为 `[*, C]` 的逻辑张量，返回形状为 `[*]` 的长整型张量。

    该函数根据输入的逻辑张量进行采样。采样方式可以是贪婪采样（argmax）、多项式采样（multinomial）、top-k 采样或 top-p 采样。

    Args:
        logits (torch.Tensor): 输入逻辑张量，形状为 `[*, C]`，其中 `C` 是候选数。
        use_sampling (bool, optional): 是否使用采样。默认为 `False`，即进行贪婪采样。
        temp (float, optional): 采样温度。默认为 `1.0`。
        top_k (int, optional): top-k 中的 `k`，用于 top-k 采样。默认为 `0`，表示不进行 top-k 采样。
        top_p (float, optional): top-p 中的 `p`，用于 top-p 采样。默认为 `0.0`，表示不进行 top-p 采样。

    Returns:
        torch.Tensor: 采样的标记，形状为 `[*]`。
    """
    # 如果使用采样且温度大于0，则应用 softmax 进行概率计算；否则，进行贪婪采样以避免除以零错误
    if use_sampling and temp > 0.0:
        # 应用温度缩放的 softmax
        probs = torch.softmax(logits / temp, dim=-1)
        if top_p > 0.0:
            # 使用 top-p 采样
            next_token = sample_top_p(probs, p=top_p)
        elif top_k > 0:
            # 使用 top-k 采样
            next_token = sample_top_k(probs, k=top_k)
        else:
            # 使用多项式采样
            next_token = multinomial(probs, num_samples=1)
    else:
        # 进行贪婪采样（argmax）
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    assert next_token.shape[-1] == 1
    # 返回形状为 `[*]` 的采样的标记
    return next_token[..., 0]


if __name__ == "__main__":

    # 设置随机种子以确保可重复性
    torch.manual_seed(1234)
    device = "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda:0"

    # 定义概率分布张量
    ps = torch.tensor([5.0, 2.0, 12.0, 6.0, 8.0, 1.0, 0.0, 4.0], device=device)
    # 初始化计数张量，统计每个候选被采样的次数
    cnts = torch.zeros(ps.shape, dtype=torch.long, device=device)
    # 总采样次数
    total_samples = 1000
    for _ in range(total_samples):
        # 对概率分布进行无放回的多项式采样
        vs = multinomial(ps, num_samples=1, replacement=False)
        # 增加对应候选的计数
        cnts[vs] += 1
    # 计算采样分布与原始分布的差异
    diff = cnts / cnts.sum() - ps / ps.sum()
    # 计算最大差异
    max_diff = diff.abs().max().cpu().item()
    print(ps / ps.sum())
    print(cnts / cnts.sum())
    assert max_diff < 1.5e-2
