import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init


class ResBlock(nn.Module):

    def __init__(self, layer, ltype="linear"):
        super(ResBlock, self).__init__()

   
        if ltype == "linear":
            self.fc = nn.Sequential(
                                nn.Linear(layer, layer, bias=True),
                                nn.ELU(),
                                nn.Linear(layer, layer, bias=True),
                               )
        elif ltype == "conv1":
            self.fc = nn.Sequential(
                                nn.Conv1d(layer, layer, kernel_size=3,padding=1),
                                nn.ELU(),
                                nn.Conv1d(layer, layer, kernel_size=3,padding=1),
                               )

    def forward(self, x):
        
        h = self.fc(x)
        return F.elu(x + h)


class Head(nn.Module):

    def __init__(self, latent_dim, output_dim_in, output_dim_out):
        super(Head, self).__init__()
        
        h_layer = 512
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h_layer),
            nn.ELU(),
            nn.Linear(h_layer, h_layer),
        )
        
        # Q function net
        self.W = nn.Sequential(
            nn.Linear(h_layer, h_layer),
            nn.ELU(),
            nn.Linear(h_layer, output_dim_in * output_dim_out)
        )

        self.b = nn.Sequential(
            nn.Linear(h_layer, h_layer),
            nn.ELU(),
            nn.Linear(h_layer, output_dim_out)
        )
        self.s = nn.Sequential(
            nn.Linear(h_layer, h_layer),
            nn.ELU(),
            nn.Linear(h_layer, output_dim_out)
        )

        self.init_layers()

    def forward(self, x):

        z = self.net(x)

        w = self.W(z).view(self.output_dim_in, self.output_dim_out)
        b = self.b(z)
        s = 1. + self.s(z)

        return w, b, s
   
    def init_layers(self):
        for b in self.b.modules():
            if isinstance(b, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                torch.nn.init.zeros_(b.weight)  

#TODO seperete the hyper net with the regular net
class Hyper_Policy(Policy):
    # Hyper net that create weights from the task for a net that estimates function pi(A|S)
    def __init__(self, task_dim, state_dim, action_dim, num_hidden, var_tasks=1.0 , init_std=1.0, min_std=1e-6):
        super(Hyper_Policy, self).__init__(input_size=state_dim, output_size=action_dim)
        
        self.min_log_std = math.log(min_std)
        self.sigma = nn.Parameter(torch.Tensor(action_dim))
        self.sigma.data.fill_(math.log(init_std))

        self.emb_dim = 64
        f_layer = 512
        self.num_layers_g = num_hidden + 2
        self.z_dim = 512
        latent_layer = self.z_dim * (self.num_layers_g)
        self.g_layer = 32

        self.hyper = nn.Sequential(
        #    nn.Linear(task_dim, self.emb_dim),
        #    nn.Tanh(),
			nn.Linear( self.emb_dim, f_layer, bias=True),
			nn.ELU(),
			ResBlock(f_layer),
			ResBlock(f_layer),
			nn.Linear(f_layer, latent_layer, bias=True),
		)

        #policy net
        self.layer1 = Head(self.z_dim,self.emb_dim, self.g_layer)
        self.hidden = nn.ModuleList(Head(self.z_dim,self.g_layer,self.g_layer) for i in range(num_hidden))
        self.mu = Head(self.z_dim,self.g_layer, action_dim)
        
        self.state_emb = nn.Linear(state_dim, self.emb_dim)
    
        self.init_layers()

    def forward(self, state, task, params=None):

		# f heads
        z = self.hyper(task)

        if len(z.shape) == 1:
            z = z.reshape(1,self.num_layers_g,-1)
        

		#action embedding
        emb = torch.tanh(self.state_emb(state))
   
        # g first layer
        w ,b ,s = self.layer1(z[:,0,:])
        out = F.elu(torch.matmul(emb, w) * s + b)
   
        # g hidden layers
        for i, layer in enumerate(self.hidden):
            w, b, s = self.hidden[i](z[:,i + 1,:])
            out = F.elu(torch.matmul(out, w) * s + b)

        # g final layer
        w, b, s = self.mu(z[:,-1,:])
        mu = torch.matmul(out, w) * s + b  

        scale = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

    def init_layers(self):
        # init f with fanin
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)  

