from typing import Union, List, Tuple

from tqdm import tqdm

import numpy as np

import torch
from torch import nn

from .utils import count_params, DimUniversalStandardScaler, kl_div as _kl_div
from .models import TimeDiffusionProjector, TimeDiffusionAttention, TimeDiffusionModel


class TD(nn.Module):
    """
    TD类提供了一个方便的框架，用于有效地使用TimeDiffusionProjector，包含所有必要的函数。
    """

    def __init__(self, base_model_init: TimeDiffusionModel = TimeDiffusionProjector,
                 verbose: bool = False, seed=42, *args, **params):
        """
        初始化方法

        参数:
            `base_model_init` - base model init func, should inherited from TimeDiffusionModel
            `verbose` - 是否输出模型参数的数量
            `seed` - 模型参数初始化的随机种子
            `input_dims` - [channels, *dims]，用于动态网络构建，最好将其作为`x.shape`（不包括批次）来传递
            `max_deg_constraint` - 限制网络规模的约束条件，如果太小会导致模型质量下降，网络中的时间块数量将取决于(1 + max_deg_constraint)
            `conv_filters` - 每个层的卷积滤波器数量
            `base_dropout` - 第一个时间块的dropout率
        """
        super().__init__()  # 调用父类的构造函数
        torch.random.manual_seed(seed)  # 设置随机种子用于重现性
        self.model = base_model_init(*args, **params)  # 初始化基础模型
        self.input_dims = self.model.input_dims  # 获取基础模型的输入维度
        self.is_fitted = False  # 设置模型未进行拟合
        if verbose:
            print(f"Created model with {count_params(self):.1e} parameters")  # 打印模型的参数数量

    def dtype(self):
        return next(self.model.parameters()).dtype

    def device(self):
        return next(self.model.parameters()).device

    def fit(self, example: Union[np.ndarray, torch.Tensor], mask: Union[None, np.ndarray, torch.Tensor] = None,
            epochs: int = 20, batch_size: int = 2, steps_per_epoch: int = 32,
            early_stopping_epochs: Union[None, int] = None,
            lr: float = 4e-4, distance_loss: Union[str, nn.Module] = "MAE",
            distribution_loss: Union[str, nn.Module] = "kl_div", distrib_loss_coef=1e-2,
            verbose: bool = False, seed=42) -> List[float]:
        """
        用于训练模型的扩散过程

        参数:
            `example` - 格式为(通道数, *尺寸)的序列、图像或视频数据
            `mask` - 如果为None，则对`example`进行完全拟合，如果与`example`具有相同的形状，则表示不拟合的点使用1进行掩码
            `epochs` - 训练轮数
            `batch_size` - 每个训练步骤使用的随机噪声数量，平衡时间和内存之间的关系
            `steps_per_epoch` - 每个轮次训练的扩散步数
            `early_stopping_epochs` - 是否在每个轮次后进行验证，并在质量降低而没有改善的情况下停止模型
            `lr` - 学习率
            `distance_loss` - 用于拟合输入示例的主要损失函数，可以是"MAE"、"MSE"或产生某种距离损失但不进行维度缩减的PyTorch nn.Module
            `distribution_loss` - 附加损失函数，可以是"kl_div"（使用内置的Kullback–Leibler散度）或产生某种分布损失但不进行维度缩减的PyTorch nn.Module
            `distrib_loss_coef` - 总损失中分布损失的比例
            `verbose` - 是否输出训练进度
            `seed` - fit方法的随机种子，用于重现性

        返回:
            训练损失列表（每个轮次每个步骤的损失）
        """
        # 距离损失函数定义
        # 判断距离损失函数是否为字符串
        if isinstance(distance_loss, str):
            # 判断距离损失函数是否在可选范围内
            if distance_loss not in ("MAE", "MSE"):
                raise NotImplementedError(f"Distance loss {distance_loss} doesn't exist")  # 抛出错误，提示距离损失函数不存在
            _mae = lambda x, y: (x - y).abs()  # 定义MAE损失函数
            _mse = lambda x, y: ((x - y) ** 2)  # 定义MSE损失函数
            distance_loss = {"MAE": _mae, "MSE": _mse}[distance_loss]  # 根据用户选择的损失函数字符串选择对应的损失函数
        # 判断距离损失函数是否不是字符串且不是nn.Module类型
        elif not isinstance(distance_loss, nn.Module):
            # 抛出错误，提示距离损失函数应为MAE、MSE或nn.Module类型，而不是当前类型
            raise NotImplementedError(f"Distance loss should be 'MAE', 'MSE' or nn.Module, got {type(distance_loss)}")

        # 分布损失函数定义
        if isinstance(distribution_loss, str):
            if distribution_loss != "kl_div":
                raise NotImplementedError(f"Distribution loss {distribution_loss} doesn't exist")
            distribution_loss = _kl_div  # 使用内置的Kullback–Leibler散度作为分布损失函数
        elif not isinstance(distribution_loss, nn.Module):
            # 抛出错误，提示分布损失函数应为"kl_div"或nn.Module类型，而不是当前类型
            raise NotImplementedError(
                f"Distribution loss should be 'kl_div' or nn.Module got {type(distribution_loss)}")

        # 检查掩码
        # 判断掩码是否不为None且形状与示例不同
        if mask is not None and mask.shape != example.shape:
            # 抛出错误，提示掩码应为None或与示例具有相同的形状，而当前形状为example.shape，掩码形状为mask.shape
            raise ValueError(
                f"Mask should None or the same shape as example, got {example.shape = } and {mask.shape = }")

        # 标准化
        self.scaler = DimUniversalStandardScaler()  # 创建标准化器对象
        train_tensor = torch.tensor(example, dtype=self.dtype(), device=self.device()).unsqueeze(0)  # 将示例转换为张量，并添加一个维度
        train_tensor = self.scaler.fit_transform(train_tensor)  # 使用标准化器对张量进行标准化处理
        X = train_tensor.repeat(batch_size, *[1] * (len(train_tensor.shape) - 1))  # 将标准化后的张量复制多份以构建训练数据集X

        if mask is not None:
            mask_tensor = ~ torch.tensor(mask, dtype=torch.bool, device=self.device()).unsqueeze(
                0)  # 将掩码转换为布尔张量，并添加一个维度
            mask_tensor = mask_tensor.repeat(batch_size, *[1] * (len(mask_tensor.shape) - 1))  # 将掩码张量复制多份以构建掩码数据集mask

        optim = torch.optim.Adam(self.parameters(), lr=lr)  # 创建Adam优化器对象
        losses = []  # 记录损失函数的值
        if early_stopping_epochs is not None:
            val_losses = []  # 记录验证集上的损失函数的值
            val_noise = torch.rand(*X.shape, device=self.device(), dtype=self.dtype())  # 创建与训练数据集X相同形状的随机噪声张量

        torch.random.manual_seed(seed)  # 设置随机种子
        for epoch in (tqdm(range(1, epochs + 1)) if verbose else range(1, epochs + 1)):
            self.model.train()
            # 创建与训练数据集X相同形状的随机噪声张量
            noise = torch.rand(*X.shape, device=self.device(), dtype=self.dtype())
            # noise_level = torch.rand(X.shape).to(device=self.device(), dtype=self.dtype())
            # noise *= noise_level
            # scaling random noise with noise level gives additional training diversity and stability in some cases
            # TODO: further research in this area

            for step in range(steps_per_epoch):
                optim.zero_grad()  # 清空优化器的梯度
                y_hat = self.model(noise)  # 使用模型对噪声进行预测
                # noise - y_hat -> X
                loss = distance_loss(noise - y_hat, X) + distrib_loss_coef * distribution_loss(y_hat, noise)  # 计算损失函数
                loss = loss.mean() if mask is None else loss[mask_tensor].mean()  # 如果存在掩码，则只计算掩码部分的平均损失
                loss.backward()  # 反向传播计算梯度
                optim.step()  # 更新模型参数

                with torch.no_grad():
                    noise -= y_hat  # 更新噪声张量
                losses.append(loss.item())  # 记录损失函数的值

            # validation
            if early_stopping_epochs is not None:
                self.model.eval()  # 设置模型为评估模式
                with torch.no_grad():
                    cur = val_noise.clone()  # 克隆验证噪声张量
                    for step in range(steps_per_epoch):
                        cur -= self.model(cur)  # 使用模型对验证噪声进行预测并更新验证噪声张量
                    val_losses.append(distance_loss(cur, X).mean().item())  # 计算验证集上的损失函数，并记录其值

                best_val_epoch = np.argmin(val_losses)  # 找到验证集损失函数最小值所对应的轮次
                if epoch - best_val_epoch - 1 >= early_stopping_epochs:
                    # 如果当前轮次减去最佳验证集轮次大于等于提前停止轮次
                    if verbose:
                        print(f"Due to early stopping fitting stops after {epoch}")  # 输出提前停止的提示信息
                        print(
                            f"Val quality of {best_val_epoch} epoch {val_losses[best_val_epoch]: .1e}")  # 输出验证集上最佳轮次的损失函数值
                        print(f"\tof current {val_losses[- 1]: .1e}")  # 输出当前轮次的验证集损失函数值
                    break

        # saving some training parameters, could be useful in inference
        self.training_steps_per_epoch = steps_per_epoch  # 保存每个epoch中的训练步数
        self.training_example = example  # 保存一个训练示例
        self.distance_loss = distance_loss  # 保存距离损失函数
        self.distribution_loss = distribution_loss  # 保存分布损失函数
        self.is_fitted = True  # 标记模型是否已经训练完成

        return losses

    @torch.no_grad()
    def restore(self, example: Union[None, np.ndarray, torch.Tensor] = None,
                shape: Union[None, List[int], Tuple[int]] = None,
                mask: Union[None, np.ndarray, torch.Tensor] = None, steps: Union[None, int] = None,
                seed: int = 42, verbose: bool = False) -> torch.Tensor:
        """
        recreates data using fitted model

        either `example` or `shape` should be provided

        3 possible workflows

            1 case of `shape`: model starts with random noise

            2 case of `example`: ignores `shape` and model starts with `example`

            3 case of `example` and `mask`: same as 2 case, but masked values persistent through diffusion process

        args:

            `example` - None or in format [channels, *dims], channels should be the same as in training example

            `shape` - None or in format [channels, *dims], channels should be the same as in training example

            `mask` - None or in format of `example`, zeros in positions, that needed to be persistent

            `steps` - steps for diffusion process, if None uses same value as in fit method

            `seed` - random seed, only necessary in case of providing only `shape`

            `verbose` - whether to output progress of diffusion process or not

        returns:
            result of diffusion process (torch.Tensor)
        """
        if not self.is_fitted:
            raise RuntimeError("Model isn't fitted")

        if example is None:
            if shape is None:
                raise ValueError("Either `example` or `shape` should be passed")

            torch.random.manual_seed(seed)
            X = torch.rand(*shape).to(device=self.device(), dtype=self.dtype())

            # no real meaning behind masking random noise
            # maybe for fun, but setting it here as None for stability
            mask = None
        else:
            if len(self.input_dims) != len(example.shape):
                raise ValueError(f"Model fitted with {len(self.input_dims)} dims, but got {len(example.shape)}")

            if self.input_dims[0] != example.shape[0]:
                raise ValueError(f"Model fitted with {self.input_dims[0]} channels, but got {example.shape[0]}")

            X = torch.tensor(example, device=self.device(), dtype=self.dtype())
            X = self.scaler.transform(X)

            if mask is not None:
                if mask.shape != example.shape:
                    raise ValueError(f"Mask should be same shape as example, got {example.shape = } {mask.shape = }")

                mask = torch.tensor(mask, device=self.device(), dtype=torch.bool)

            # provided example could have nan values
            nan_mask = torch.isnan(X)
            X[nan_mask] = torch.randn(nan_mask.sum(), device=X.device, dtype=X.dtype)

        steps = self.training_steps_per_epoch if steps is None else steps
        self.model.eval()
        for step in (tqdm(range(steps)) if verbose else range(steps)):
            preds = self.model(X)
            if mask is None:
                X -= preds
            else:
                X[mask] -= preds[mask]

        X = self.scaler.inverse_transform(X)
        return X

    def forecast(self, horizon: Union[int, Tuple[int], List[int]], steps: Union[None, int] = None,
                 seed: int = 42, verbose: bool = False) -> torch.Tensor:
        """
        convinient version of `restore` to only get prediction for forecasting `horizon`
        uses trained sequence as masked (persistent) reference to forecast next values

        WORKS ONLY FOR 1D DATA

        args:

            `horizon` - forecasting horizon (e.g. number of values to forecast)

            `steps` - steps for diffusion process, if None uses same value as in fit method

            `seed` - random seed for reproducability

            `verbose` - whether to output progress of diffusion process or not

        returns:
            forecasted values (torch.Tensor in shape [channels, horizon])
        """
        if not self.is_fitted:
            raise RuntimeError("Model isn't fitted")
        if len(self.input_dims) != 2:
            raise NotImplementedError("forecast method works only for 1D data")

        np.random.seed(seed)
        example = np.append(self.training_example, np.zeros((self.input_dims[0], horizon)), axis=1)
        mask = np.zeros_like(example)
        mask[:, - horizon:] = 1

        res = self.restore(example=example, mask=mask, steps=steps, seed=seed, verbose=verbose)
        return res[:, - horizon:]

    @torch.no_grad()
    def synth(self, start=None, proximity: float = 0.9, samples: int = 8, batch_size: int = 8,
              step_granulation: int = 10,
              seed: int = 42, verbose: bool = False) -> torch.Tensor:
        """
        根据与原始示例的接近程度生成合成数据
            如果为None，则可以从随机噪声开始并去噪到一定程度
            或者从提供的样本[start, channesl, *dims]开始工作

        参数:

            `start` - 扩散过程从
                当为None时，从随机噪声开始
                当为其他情况时，从提供的样本[start, channesl, *dims]开始

            `proximity` - 合成样本与拟合示例的相似度
                应在[0.0, 1.0]范围内，其中0.0表示随机噪声，1.0表示完全恢复

            `samples` - 要生成的合成样本数量

            `batch_size` - 在一个扩散过程中生成的序列数量
                设置速度和生成内之间的权衡

            `step_granulation` - 将通常步骤分成的子步骤数量
                显着增加计算时间，以换取合成样本与所选接近度的更好接近程度

            `seed` - 用于可重复性的随机种子

            `verbose` - 是否输出进度

        返回:
            torch.Tensor，形状为[samples, channels, *dims]
    """
        # 检查模型是否已经拟合
        if not self.is_fitted:
            raise RuntimeError("Model isn't fitted")
        # 检查proximity是否在[0, 1]范围内
        if not 0. <= proximity <= 1.:
            raise ValueError(f"proximity should be in [0, 1], got {proximity = }")
        dist = _kl_div  # 设置距离度量函数和粒度系数
        gran_coef = 1 / step_granulation
        torch.random.manual_seed(seed)  # 设置随机种子

        # 估计已拟合模型的接近程度
        # TODO: 考虑更好的接近程度机制
        if start is None:
            # 如果没有提供起始样本，则生成随机噪声作为起始样本
            x = torch.rand(*self.input_dims, device=self.device(), dtype=self.dtype())
        else:
            # 如果提供了起始样本，则使用提供的样本作为起始样本
            x = torch.tensor(start[0], device=self.device(), dtype=self.dtype())
        # 将拟合示例转换为numpy数组并计算与起始样本的距离分数
        ref = self.training_example.numpy() if type(self.training_example) is torch.Tensor else self.training_example
        scores = [dist(x.cpu().numpy(), ref).mean()]
        # 打印拟合接近程度的估计信息
        if verbose:
            print("Estimating fitted proximity...")

        # 将模型设置为评估模式
        self.model.eval()
        # 计算迭代次数
        _range = range(self.training_steps_per_epoch * step_granulation)
        # 在迭代过程中更新起始样本并计算距离分数
        for _ in (tqdm(_range) if verbose else _range):
            preds = self.model(x) * gran_coef  # 使用模型生成预测
            x -= preds  # 更新起始样本
            scores.append(dist(x.cpu().numpy(), ref).mean())  # 计算与拟合示例的距离分数

        # 对距离分数进行归一化处理
        scores = 1 - np.array(scores)  # 将距离转换为相似度
        scores = (scores - scores.min()) / (scores.max() - scores.min())  # 归一化距离分数
        best_step = np.argmin(np.abs(scores - proximity))  # 找到与目标相似度最接近的粒度步骤
        if verbose:
            print(f"Best granulated step is {best_step}")  # 打印最佳的粒度步骤

        # 生成新样本
        res = []
        # 打印生成过程信息
        if verbose:
            print("Generating...")
        # 设置循环范围和步长
        _range = range(0, samples, batch_size)
        # 在循环中生成样本
        for i in (tqdm(_range) if verbose else _range):
            if start is None:
                # 如果没有提供起始样本，则生成随机噪声作为起始样本
                x = torch.rand(min(batch_size, samples - i),
                               *self.input_dims, device=self.device(), dtype=self.dtype())
            else:
                # 如果提供了起始样本，则使用提供的样本作为起始样本
                x = torch.tensor(start[i: i + batch_size], device=self.device(), dtype=self.dtype())
            # 在最佳的粒度步骤之前进行迭代更新
            for _ in range(best_step - 1):
                x -= self.model(x) * gran_coef
            # 将生成的样本添加到结果列表中
            res.append(x)
        # 将结果列表转换为torch张量
        res = torch.concat(res, dim=0)
        # 对生成的样本进行逆缩放操作
        res = self.scaler.inverse_transform(res)

        return res
