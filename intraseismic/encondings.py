import torch
import numpy as np
import math
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def exp_enc(x_input, levels):
    result_list = []
    for i in range(levels):
        # temp = 2.0**i * torch.pi * x_input   
        temp = torch.pi * (i+1)/2 * torch.pi * x_input
        result_list.append(torch.sin(temp)) 
        result_list.append(torch.cos(temp)) 

    result_list = torch.cat(result_list, dim=-1) 
    return result_list

def rff_enc(x_input, levels, scale, dim=2):
    B = scale * torch.randn(size=(dim, levels), requires_grad=False)
    proj = (2. * torch.pi * x_input) @ B
    result_list = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    return result_list


def rbf_enc(x_input, levels, sigma):
    dic1 = torch.empty((1, levels)).to(device)
    for i in range(levels):
        dic1[0,i] = 2*i/(levels*2)
        
    rbfemb1 = (-0.5*(x_input[:,0].unsqueeze(1)-dic1)**2/(sigma**2)).exp()
    rbfemb2 = (-0.5*(x_input[:,1].unsqueeze(1)-dic1)**2/(sigma**2)).exp()
    rbfemb = torch.cat([rbfemb1,rbfemb2],1)
    rbfemb = rbfemb/(rbfemb.norm(dim=1).max())
    return rbfemb

def rbf_enc3D(x_input, levels, sigma):
    dic1 = torch.empty((1, levels)).to(device)
    for i in range(levels):
        dic1[0,i] = 2*i/(levels*2)
        
    rbfemb1 = (-0.5*(x_input[:,0].unsqueeze(1)-dic1)**2/(sigma**2)).exp()
    rbfemb2 = (-0.5*(x_input[:,1].unsqueeze(1)-dic1)**2/(sigma**2)).exp()
    rbfemb3 = (-0.5*(x_input[:,2].unsqueeze(1)-dic1)**2/(sigma**2)).exp()
    rbfemb = torch.cat([rbfemb1,rbfemb2, rbfemb3],1)
    rbfemb = rbfemb/(rbfemb.norm(dim=1).max())
    return rbfemb


