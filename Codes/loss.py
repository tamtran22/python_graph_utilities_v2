import torch
from torch.nn.modules.loss import _Loss
import torch.Tensor as Tensor
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from typing import Optional
from torch import _reduction as _Reduction

class _LossFunction():
    def weight_mean_squared_loss(self, 
        input: Tensor, 
        target: Tensor, 
        weight: Tensor=None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str='mean'
    ) -> Tensor:
        if not (target.size() == input.size()):
            pass
        
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)

            


class WeightedMSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction:str='mean'):
        super().__init__(size_average, reduce, reduction)
    def forward(self, input : Tensor, target : Tensor, weight : Tensor):
        return _LossFunction.weight_mean_squared_loss(input, target, weight, reduction=self.reduction)

class PINNLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction:str='mean'):
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, input : Tensor, target : Tensor):