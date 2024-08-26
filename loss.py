import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

class ObjectiveLoss(nn.Module):
    """
    Our objective consists of two parts, the first part is to train an optimal posterior factor model, and the second part is to effectively guide the leaning of factor predictor by the posterior factors.
    Thus, the loss function of model is

        L(x, y) = −(1/N) · ∑ log(P_dec(y_hat_i = y_i|x, z_post)) + γ · KL(P_enc(z|x, y), P_pred(z|x))
    
    where the first loss term is the negative log likelihood, to reduce the reconstruction error of posterior factor model, and y_hat_i = α_i + β_i · z_post is the reconstructed return of i-th stock.The second loss term is the Kullback–Leibler divergence (KLD) between the distribution of prior and posterior factors, for enforcing the prior factors to approximate to the posterior factors, γ is the weight of KLD loss.
    """
    def __init__(self, 
                 gamma=1, 
                 recon_loss_type:Literal["MSE", "NLL"]="MSE") -> None:
        super().__init__()
        self.gamma = gamma
        if recon_loss_type == "MSE":
            self.recon_loss = MSE_Loss()
        elif recon_loss_type == "NLL":
            self.recon_loss = NLL_Loss()
        self.kl_div_loss = KL_Div_Loss()

    def forward(self, 
                y, 
                y_hat, 
                mu_prior, 
                sigma_prior, 
                mu_posterior, 
                sigma_posterior) -> torch.Tensor:

        # Reconstruction Loss
        recon_loss = self.recon_loss(y_hat, y)
        
        # KL Div Loss
        kld_loss = self.kl_div_loss(mu_prior, sigma_prior, mu_posterior, sigma_posterior)
        
        return recon_loss + kld_loss, recon_loss, kld_loss   
    

class KL_Div_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, 
                mu_prior, 
                sigma_prior, 
                mu_posterior, 
                sigma_posterior) -> torch.Tensor:
        # kl_divergence between two gaussian distribution:
        # KL(N(x;μ_1, σ_1)||N(x;μ_2, σ_2)) = log（σ_2/σ_1) + (σ_1^2 + (μ_1 - μ_2)^2) / (2·σ_2^2) - 1/2
        return torch.sum(torch.log(sigma_prior / sigma_posterior) + (sigma_posterior**2 + (mu_posterior - mu_prior)**2) / (2 * sigma_prior**2) - 0.5)

class MSE_Loss(nn.MSELoss):
    def __init__(self,
                 scale = 100,
                 size_average=None, 
                 reduce=None, 
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.scale = scale
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.scale * super().forward(pred, target)

class NLL_Loss(nn.CrossEntropyLoss):
    def __init__(self, 
                 weight: torch.Tensor | None = None, 
                 size_average=None, 
                 ignore_index: int = -100, 
                 reduce=None, 
                 reduction: str = 'mean', 
                 label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(pred, target)
    
class PearsonCorr(nn.Module):
    def __init__(self):
        super(PearsonCorr, self).__init__()

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        cov = torch.mean((pred - pred_mean) * (target - target_mean))
        pred_std = torch.std(pred)
        target_std = torch.std(target)
        corr = cov / (pred_std * target_std)
        return corr

class SpearmanCorr(nn.Module):
    def __init__(self):
        super(SpearmanCorr, self).__init__()

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        pred_rank = pred.argsort().argsort().float()
        target_rank = target.argsort().argsort().float()

        pred_mean = torch.mean(pred_rank)
        target_mean = torch.mean(target_rank)
        cov = torch.mean((pred_rank - pred_mean) * (target_rank - target_mean))
        pred_std = torch.std(pred_rank)
        target_std = torch.std(target_rank)
        corr = cov / (pred_std * target_std) 
        return corr
