from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(model_or_params, optimizer: str, lr: float, weight_decay: float = 0.01, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
    """
    获取优化器，支持模型对象或参数列表
    Args:
        model_or_params: 模型对象(nn.Module)或参数列表(list)
        optimizer: 优化器名称
        lr: 学习率  
        weight_decay: 权重衰减
        betas: Adam优化器的beta参数 (beta1, beta2)
        eps: Adam优化器的epsilon参数
    """
    wd_params, nwd_params = [], []
    
    # 统一处理参数获取
    if hasattr(model_or_params, 'parameters') and callable(getattr(model_or_params, 'parameters')):
        # 如果是模型对象，调用.parameters()方法
        param_iterator = model_or_params.parameters()
    elif isinstance(model_or_params, (list, tuple)):
        # 如果是参数列表或元组
        param_iterator = model_or_params
    else:
        raise ValueError(f"Expected model with .parameters() method or parameter list, got {type(model_or_params)}")
    
    for p in param_iterator:
        if hasattr(p, 'requires_grad') and p.requires_grad:
            if p.dim() == 1:
                nwd_params.append(p)
            else:
                wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)