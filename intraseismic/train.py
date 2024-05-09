import time
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from intraseismic.utils import batch_tv

import matplotlib.pyplot as plt

def train_is_mb(dataloader, dims_model,
                  G, Gl,                  
                  net, optimizer,  
                  alpha_mtv,
                  beta,
                  l2 = False):
    """
    IntraSeismic train function for a mini-batches (mb) of data.

    This function performs a single training step of IntraSeismic that predicts a subsurface
    model of rock elastic properties.


    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        A DataLoader object that yields batches of data: coordinates, seismic data and the background 
        model. The DataLoader should output a tuple of (coords, d, mback)
    d : torch.Tensor
        The seismic data. 
        Expected shape: (N,) where N is the number of samples.
    dims_model : tuple
        The dimensions of the model or data.
        Expected shape: (D, H, W) for 3D or (H, W) for 2D datasets, D is the depth, H is the height, 
        and W is the width.
    G : Operator
        The seismic modeling operator. Should be compatible with the model's dimensions.
    Gl : Operator
        The seismic modeling operator for the last batch.
    net : torch.nn.Module
        The IntraSeismic network. 
        Expected to output a tensor of shape (N,).
    optimizer : torch.optim.Optimizer
        The optimizer used to update the parameters of IntraSeismic.
    alpha_mtv : float
        The scaling factor for the predicted model total variation loss.
    beta : float
        The scaling factor for the L1 norm of the output of net.
    l2 : bool, optional
        Use L2 norm as regularization of the ouput of the net instead of L1 norm.        

    Returns
    -------
    float
        The total loss computed for this training step.
    float
        The duration of the training step in seconds.
    """
                
    start_time = time.time()
    mse_loss = torch.nn.MSELoss()
    total_loss = 0.0
    for i, (coords, d, mback) in enumerate(dataloader):
        coords = coords.reshape(-1, len(dims_model)).to(device)
        d = d.reshape(-1).to(device)
        mback = mback.reshape(-1).to(device)
        
        delta_model = net(coords).float().view(-1)
        model = delta_model + mback
        
        # To handle different batch sizes.
        try:
            dpred = G.apply(model).float().to(device)        
        except ValueError:
            dpred = Gl.apply(model).float().to(device)

        if len(dims_model)==2:
            n = d.shape[0] // (dims_model[0] * dims_model[1]) # size of batch

        elif len(dims_model)==3:
            n = d.shape[0] // (dims_model[0] * dims_model[1] * dims_model[2]) # size of batch
        loss = mse_loss(dpred, d)    
        loss += alpha_mtv * batch_tv(model.view(n, *dims_model))
        
        if l2 == False:
            loss += beta * torch.mean(torch.abs(delta_model))
        elif l2 == True:
            loss += beta * torch.mean(delta_model**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_duration = time.time() - start_time
    return total_loss / len(dataloader), train_duration



def train_is_sb(coords, d, mback, dims_model, G,
                net, optimizer,
                alpha_mtv,
                beta,
                l2 = False):
    """
    IntraSeismic train function for a single batch (sb) of data.

    This function performs a single training step of IntraSeismic that predicts a subsurface
    model of rock elastic properties.


    Parameters
    ----------
    coords : torch.Tensor
        The input coordinates 
        Expected shape: (N, *) where '*' represents the spatial dimensions (N, Z, X) or (N, Z, Y, X)
    d : torch.Tensor
        The seismic data. 
        Expected shape: (N,) where N is the number of samples.
    mback : torch.Tensor
        The background model.
        Expected shape: (N,) where N is the number of samples.
    dims_model : tuple
        The dimensions of the model or data.
        Expected shape: (D, H, W) for 3D or (H, W) for 2D datasets, D is the depth, H is the height, 
        and W is the width.
    G1 : Operator
        The seismic modeling operator. Should be compatible with the model's dimensions.
    net : torch.nn.Module
        The IntraSeismic network. 
        Expected to output a tensor of shape (N,).
    optimizer : torch.optim.Optimizer
        The optimizer used to update the parameters of IntraSeismic.
    alpha_mtv : float
        The scaling factor for the predicted model total variation loss.
    beta : float
        The scaling factor for the L1 norm of the output of net.

    Returns
    -------
    float
        The total loss computed for this training step.
    float
        The duration of the training step in seconds.
    """
                 
    start_time = time.time()
    mse_loss = torch.nn.MSELoss()

    delta_model = net(coords).float().view(-1)
    model = delta_model + mback
    dpred = G.apply(model).float().to(device)

    loss = mse_loss(dpred, d)
    loss += alpha_mtv * batch_tv(model.view(1, *dims_model))
    
    if l2 == False:
        loss += beta * torch.mean(torch.abs(delta_model))
    elif l2 == True:
        loss += beta * torch.mean(delta_model**2)
              
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_duration = time.time() - start_time
    return loss.item(), train_duration


def train_is_hd(coords, d, mback, xwell, mwell, dims_model, G,
                net, optimizer,
                alpha_mtv,
                beta,
                use_mback = False):
    """
    IntraSeismic train function for a single batch (sb) of data using well data from 1 well as 
    a hard constraint (only tested in 2D data).

    Parameters
    ----------
    coords : torch.Tensor
        The input coordinates 
        Expected shape: (N, *) where '*' represents the spatial dimensions (N, Z, X) or (N, Z, Y, X)
    d : torch.Tensor
        The seismic data. 
        Expected shape: (N,) where N is the number of samples.
    mback : torch.Tensor
        The background model.
        Expected shape: (N,) where N is the number of samples.
    xwell : torch.Tensor
        Array (Z,) containing the X coordinates of the well. 

    mwell : torch.Tensor
        Array (Z,) containing the impedance log of the well.    
    dims_model : tuple
        The dimensions of the model or data.
        Expected shape: (D, H, W) for 3D or (H, W) for 2D datasets, D is the depth, H is the height, 
        and W is the width.
    G : Operator
        The seismic modeling operator. Should be compatible with the model's dimensions.
    net : torch.nn.Module
        The IntraSeismic network. 
        Expected to output a tensor of shape (N,).
    optimizer : torch.optim.Optimizer
        The optimizer used to update the parameters of IntraSeismic.
    alpha_mtv : float
        The scaling factor for the predicted model total variation loss.
    beta : float
        The scaling factor for the L1 norm of the output of net.

    Returns
    -------
    float
        The total loss computed for this training step.
    float
        The duration of the training step in seconds.
    """
    
    start_time = time.time()
    mse_loss = torch.nn.MSELoss()

    delta_model = net(coords).float().view(-1)
    model = (coords[:,1] - xwell.repeat_interleave(dims_model[1])) * delta_model + mwell.repeat_interleave(dims_model[1])
    dpred = G.apply(model).float().to(device)

    loss = mse_loss(dpred, d)
    loss += alpha_mtv * batch_tv(model.view(1, *dims_model))
    
    if use_mback == False:
        loss += beta * torch.mean(delta_model**2)
    elif use_mback == True:
        loss += beta * torch.mean((model-mback)**2)
              
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_duration = time.time() - start_time
    return loss.item(), train_duration



def train_is_prestack(coords, d, mback, ntheta, dims_model, G,
                      net, optimizer,
                      alphas,
                      betas):
    """
    IntraSeismic train function for pre-stack seismic inversion with a single batch (sb) of seismic data.
    
    Parameters
    ----------
    coords : torch.Tensor
        The input coordinates 
        Expected shape: (N, *) where '*' represents the spatial dimensions (N, Z, X) or (N, Z, Y, X)
    d : torch.Tensor
        The pre-stack seismic data. 
        Expected shape: (Z, ntheta, X).
    mback : torch.Tensor
        The background model.
        Expected shape: (Z, 3, X).
    ntheta : int
        Number of angles in the pre-stack seismic data.  
    dims_model : tuple
        The dimensions of the model or data.
        Expected shape: (D, H, W) for 3D or (H, W) for 2D datasets, D is the depth, H is the height, 
        and W is the width.
    G : Operator
        The pre-stack seismic modeling operator. Should be compatible with the model's dimensions.
    net : torch.nn.Module
        The IntraSeismic network. 
        Expected to output a tensor of shape (N,).
    optimizer : torch.optim.Optimizer
        The optimizer used to update the parameters of IntraSeismic.
    alphas : np.array
        The scaling factor for the total variation of the 3 predicted models.
    beta : np.array
        The scaling factor for the L1 norm of the output of the nextork for the 3 predicted models.

    Returns
    -------
    float
        The total loss computed for this training step.
    float
        The duration of the training step in seconds.
    """
    
    start_time = time.time()
    mse_loss = torch.nn.MSELoss()

    delta_model = net(coords).float().view(-1).reshape(dims_model[0], 3, dims_model[1])
    model = delta_model + mback
    dpred = G.apply(model.permute(1, 0, 2).ravel()).reshape(ntheta, *dims_model).permute(1, 0, 2).float().to(device)
    
    loss = mse_loss(dpred, d)
    loss += alphas[0] * batch_tv(model[:,0].view(1, *dims_model))
    loss += alphas[1] * batch_tv(model[:,1].view(1, *dims_model))
    loss += alphas[2] * batch_tv(model[:,2].view(1, *dims_model))
    loss += betas[0] * torch.mean(torch.abs(delta_model[:,0]))
    loss += betas[1] * torch.mean(torch.abs(delta_model[:,1]))
    loss += betas[2] * torch.mean(torch.abs(delta_model[:,2]))
         
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_duration = time.time() - start_time
    return loss.item(), train_duration


def train_is_prestack_3nets(coords, d, mback, ntheta, dims_model, G,
                            net1, net2, net3, optimizer,
                            alphas,
                        betas):
    """
    IntraSeismic (with 3 networks, one fot each inverted model) train function for pre-stack seismic 
    inversion with a single batch (sb) of seismic data.
    
    Parameters
    ----------
    coords : torch.Tensor
        The input coordinates 
        Expected shape: (N, *) where '*' represents the spatial dimensions (N, Z, X) or (N, Z, Y, X)
    d : torch.Tensor
        The pre-stack seismic data. 
        Expected shape: (Z, ntheta, X).
    mback : torch.Tensor
        The background model.
        Expected shape: (Z, 3, X).
    ntheta : int
        Number of angles in the pre-stack seismic data.  
    dims_model : tuple
        The dimensions of the model or data.
        Expected shape: (D, H, W) for 3D or (H, W) for 2D datasets, D is the depth, H is the height, 
        and W is the width.
    G : Operator
        The pre-stack seismic modeling operator. Should be compatible with the model's dimensions.
    net1 : torch.nn.Module
        The IntraSeismic network. 
        Expected to output a tensor of shape (N,).
    net2 : torch.nn.Module
        The IntraSeismic network. 
        Expected to output a tensor of shape (N,).
    net3 : torch.nn.Module
        The IntraSeismic network. 
        Expected to output a tensor of shape (N,).
    optimizer : torch.optim.Optimizer
        The optimizer used to update the parameters of IntraSeismic.
    alphas : np.array
        The scaling factor for the total variation of the 3 predicted models.
    beta : np.array
        The scaling factor for the L1 norm of the output of the nextork for the 3 predicted models.

    Returns
    -------
    float
        The total loss computed for this training step.
    float
        The duration of the training step in seconds.
    """
    
    start_time = time.time()
    mse_loss = torch.nn.MSELoss()

    delta_model1 = net1(coords).float().view(-1).reshape(*dims_model)
    delta_model2 = net2(coords).float().view(-1).reshape(*dims_model)
    delta_model3 = net3(coords).float().view(-1).reshape(*dims_model)
    
    model1 = delta_model1 + mback[0:,0]
    model2 = delta_model2 + mback[0:,1]
    model3 = delta_model3 + mback[0:,2]

    model = torch.stack([model1, model2, model3], dim=1) # Sum the background model
    
    dpred = G.apply(model.permute(1, 0, 2).ravel()).reshape(ntheta, *dims_model).permute(1, 0, 2).float().to(device)
    
    loss = mse_loss(dpred, d)
    loss += alphas[0] * batch_tv(model1.view(1, *dims_model))
    loss += alphas[1] * batch_tv(model2.view(1, *dims_model))
    loss += alphas[2] * batch_tv(model3.view(1, *dims_model))
    loss += betas[0] * torch.mean(torch.abs(delta_model1))
    loss += betas[1] * torch.mean(torch.abs(delta_model2))
    loss += betas[2] * torch.mean(torch.abs(delta_model3))
         
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_duration = time.time() - start_time
    return loss.item(), train_duration