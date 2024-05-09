import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    """
    A multilayer perceptron (MLP) neural network.

    Parameters
    ----------
    D : int, optional
        The depth of the network (number of layers), by default 8.
    W : int, optional
        The width of each layer (number of neurons), by default 256.
    output_ch : int, optional
        The number of output channels, by default 1.
    enc : {'exp', 'exp3d', 'hash', 'rff'}, optional
        The type of encoding to use for the input data, by default 'exp'.
    levels : int, optional
        The number of levels used in the encoding, by default 10.
    scale : int, optional
        The scale factor applied in the encoding, by default 10.
    """    
    
    def __init__(self,  D=8, W=256, output_ch=1, enc='exp', levels=10, scale=10):
        super(MLP, self).__init__()       
        self.D = D
        self.W = W
        self.levels = levels
        self.scale = scale
        self.enc = enc
        
        if enc=='exp':
            self.input_ch = self.levels*4
            
        elif enc=='exp3d':
            self.input_ch = self.levels*6
            
        elif enc=='hash' or enc=='rff':
            self.input_ch = self.levels*2 
            
        self.layers_linears = nn.ModuleList([nn.Linear(self.input_ch, W)]  + [nn.Linear(W, W) for i in range(D-1)])
        self.output_linear = nn.Linear(W, output_ch)
        
    def forward(self, x):
        y = x    
        for i, l in enumerate(self.layers_linears):
            y = self.layers_linears[i](y)
            y = F.relu(y)
        
        output = self.output_linear(y)
        return output  

class MLP_dropout(nn.Module):
    """
    A multilayer perceptron (MLP) neural network with dropout.

    Parameters
    ----------
    D : int, optional
        The depth of the network (number of layers), by default 8.
    W : int, optional
        The width of each layer (number of neurons), by default 256.
    output_ch : int, optional
        The number of output channels, by default 1.
    enc : {'exp', 'exp3d', 'hash', 'rff'}, optional
        The type of encoding to use for the input data, by default 'exp'.
    levels : int, optional
        The number of levels used in the encoding, by default 10.
    scale : int, optional
        The scale factor applied in the encoding, by default 10.
    dropout : float
        probability of an element to be zeroed.        
    """   
    def __init__(self,  D=8, W=256, output_ch=1, enc='hash', levels=10, scale=10, dropout=0.2):
        super(MLP_dropout, self).__init__()      
        self.D = D
        self.W = W
        self.levels = levels
        self.scale = scale
        self.enc = enc
        
        if enc=='exp':
            self.input_ch = self.levels*4
            
        elif enc=='hash':
            self.input_ch = self.levels*2
            
        self.layers_linears = nn.ModuleList([nn.Linear(self.input_ch, W)]  + [nn.Linear(W, W) for i in range(D-1)])
        self.output_linear = nn.Linear(W, output_ch)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):  
        for i, l in enumerate(self.layers_linears):
            x = self.layers_linears[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        output = self.output_linear(x)
        return output  
    

def create_model(n_input_dims, n_output_dims, config, dropout=False):
    """
    Create a sequential model (IntraSeismic) with a hash encoding layer followed by a multilayer perceptron (MLP).

    Parameters
    ----------
    n_input_dims : int
        The number of dimensions for the input data.
    n_output_dims : int
        The number of dimensions for the output data.
    config : dict
        A configuration dictionary with keys "encoding" for the encoding layer and "mlp" for the MLP layer.
    dropout : bool, optional
        A flag to determine whether to use an MLP with dropout layers. Default is False.

    Returns
    -------
    torch.nn.Sequential
        IntraSeismic model comprising the hash encoding layer and the MLP layer.
    """
    
    encoding = tcnn.Encoding(n_input_dims, config["encoding"], dtype=torch.float32)
    if dropout==False:
        mlp = MLP(D=config["mlp"]["Depth"], W=config["mlp"]["Width"], enc=config["mlp"]["enc"], 
                  levels=config["encoding"]["n_levels"], output_ch=n_output_dims)
    elif dropout==True:
        mlp = MLP_dropout(D=config["mlp"]["Depth"], W=config["mlp"]["Width"], enc=config["mlp"]["enc"], 
                  levels=config["encoding"]["n_levels"], output_ch=n_output_dims)        
    return torch.nn.Sequential(encoding, mlp)