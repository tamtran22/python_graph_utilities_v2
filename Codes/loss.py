import torch
from torch.nn.modules.loss import _Loss
import torch.Tensor as Tensor
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from typing import Optional
from torch import _reduction as _Reduction
import torch.nn.functional as F

class LossFunction():
    def weight_mean_squared_loss(self, 
        input: Tensor, 
        target: Tensor, 
        weight: Tensor=None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str='mean'
    ) -> Tensor:
        if not (target.size() == input.size()):
            raise ValueError('Input size is different from target size.')
        
    def PDE_loss(self,
        P: Tensor, # Pressure
        Q: Tensor, # Flowrate
        L: Tensor, # Length
        D: Tensor, # Diameter
        t: Tensor, # time
        rho: float = 1.12, pi: float = 3.1415926, mu: float = 1.64E-5, K: float = 1.
    ) -> Tensor:
        unsteady_loss = (4*rho*L) / (pi * F.square(D))
        kinematic_loss = (16*K*rho) / ((pi**2) * F.square(F.square(D)))
        viscous_loss = (128*mu*L) / (pi * F.square(F.square(D)))
    
    def kinematic_loss(self,
        P, Q, L, D, rho: float = 1.12, pi: float = 3.1415926, mu: float = 1.64E-5, K: float = 1.
    ):
        Kin = (16*K*rho) / ((pi**2) * F.square(F.square(D)))
        loss = Kin * Q[:,:-1]
        return F.mean(loss)
    
    def unsteady_loss(self,
        P, Q, L, D, rho: float = 1.12, pi: float = 3.1415926, mu: float = 1.64E-5, K: float = 1., dt:float=0.1
    ):
        Uns = (4*rho*L) / (pi * F.square(D))
        loss = (Uns / dt)*(Q[:,1:]-Q[:,:-1])
        return F.mean(loss)

    def viscous_loss(self,
        P, Q, L, D, rho: float = 1.12, pi: float = 3.1415926, mu: float = 1.64E-5, K: float = 1.
    ):
        Vis = (128*mu*L) / (pi * torch.square(F.square(D)))
        loss = Vis * Q[:,1:]
        return F.mean(loss)

            


# class WeightedMSELoss(_Loss):
#     def __init__(self, size_average=None, reduce=None, reduction:str='mean'):
#         super().__init__(size_average, reduce, reduction)
#     def forward(self, input : Tensor, target : Tensor, weight : Tensor):
#         return LossFunction.weight_mean_squared_loss(input, target, weight, reduction=self.reduction)

# class PINNLoss(_Loss):
#     def __init__(self, size_average=None, reduce=None, reduction:str='mean'):
#         super().__init__(size_average, reduce, reduction)
#     def forward(self, input : Tensor, target : Tensor):
#         pass