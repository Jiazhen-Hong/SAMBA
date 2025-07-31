"""
Created on Tue Nov 14 18:12:39 2024
@author: jiazhen@emotiv.com
"""

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class MSEWithSpectralLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        MSE + Spectral Loss
        Args:
            alpha (float): Weight for MSE Loss
            beta (float): Weight for Spectral Loss
        """
        super(MSEWithSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Time domain loss (MSE)
        mse = self.mse_loss(y_pred, y_true)
        
        # Frequency domain loss (Spectral Loss)
        y_true_fft = torch.fft.rfft(y_true, dim=-1)
        y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
        
        spectral_loss = torch.mean(torch.abs(y_true_fft - y_pred_fft)**2)
        
        # Combine the losses
        total_loss = self.alpha * mse + self.beta * spectral_loss
        return total_loss



class SpectralLoss(nn.Module):
    def __init__(self):
        super(SpectralLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        y_true_fft = torch.fft.rfft(y_true, dim=-1)
        y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
        spectral_loss = torch.mean(torch.abs(y_true_fft - y_pred_fft) ** 2)
        return spectral_loss
    
# class L1WithSpectralLoss(nn.Module):
#     def __init__(self, alpha=1.0, beta=1.0):
#         super(L1WithSpectralLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.l1_loss_fn = nn.L1Loss()

#         # Store each component for external logging
#         self.last_l1 = None
#         self.last_fft = None

#     def forward(self, y_pred, y_true):
#         self.last_l1 = self.l1_loss_fn(y_pred, y_true)

#         y_true_fft = torch.fft.rfft(y_true, dim=-1)
#         y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
#         self.last_fft = torch.mean(torch.abs(y_true_fft - y_pred_fft) ** 2)

#         total_loss = self.alpha * self.last_l1 + self.beta * self.last_fft
#         return total_loss

class L1WithSpectralLoss(nn.Module):
    # def __init__(self, alpha=1.0, beta=1.0, fft_scale_factor=1e-2):
    def __init__(self, alpha=1.0, beta=1.0, fft_scale_factor=1):
        super(L1WithSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.fft_scale_factor = fft_scale_factor
        self.l1_loss_fn = nn.L1Loss()

        # Store each component for external logging (before scale)
        self.last_l1 = None
        self.last_fft = None

    def forward(self, y_pred, y_true):
        # Compute L1 loss
        self.last_l1 = self.l1_loss_fn(y_pred, y_true)

        # Compute spectral loss
        y_true_fft = torch.fft.rfft(y_true, dim=-1)
        y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
        self.last_fft = torch.mean(torch.abs(y_true_fft - y_pred_fft) ** 2)

        # Apply manual scaling to FFT loss
        total_loss = (
            self.alpha * self.last_l1 +
            self.beta * self.fft_scale_factor * self.last_fft
        )
        return total_loss

class L1WithSpectralERPLoss(nn.Module):
    def __init__(self, erp_template, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.erp_template = erp_template.detach()
        self.l1_loss_fn = nn.L1Loss()

        # initial
        self.last_l1 = None
        self.last_fft = None
        self.last_erp = None

    def forward(self, y_pred, y_true):
        self.last_l1 = self.l1_loss_fn(y_pred, y_true)

        y_true_fft = torch.fft.rfft(y_true, dim=-1)
        y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
        self.last_fft = torch.mean(torch.abs(y_true_fft - y_pred_fft) ** 2)

        erp_target = self.erp_template.unsqueeze(0).expand(y_pred.shape[0], -1, -1)
        self.last_erp = self.l1_loss_fn(y_pred, erp_target)

        total = self.alpha * self.last_l1 + self.beta * self.last_fft + self.gamma * self.last_erp
        return total

