from pyro.nn import DenseNN
from utils import *



class C_Processor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[16]):
        super(C_Processor, self).__init__()
        self.net = DenseNN(input_dim, hidden_dims=hidden_dim,
                  param_dims=[output_dim], nonlinearity=nn.ReLU())

    def forward(self, x):
        return self.net(x)


class ParamPrior(nn.Module):
    def __init__(self, c_dim, latent_dim, hidden_dim=[16]):
        super(ParamPrior, self).__init__()

        self.c_processor = C_Processor(c_dim, latent_dim, hidden_dim)
        self.mu_net = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.logvar_net = nn.Linear(in_features=latent_dim, out_features=latent_dim)

    def forward(self, c):
        c = self.c_processor(c)
        return self.mu_net(c), self.logvar_net(c)


class Syn_Encoder(nn.Module):
    def __init__(self, data_dim, c_dim, latent_dim, hidden=[16]):
        super(Syn_Encoder, self).__init__()


        self.c_processor = C_Processor(c_dim, latent_dim, hidden)
        self.net = DenseNN(latent_dim + data_dim, hidden_dims=hidden, param_dims=[latent_dim], nonlinearity=nn.ReLU())
        self.nc_net = DenseNN(data_dim, hidden_dims=hidden, param_dims=[latent_dim], nonlinearity=nn.ReLU())

        self.mu_c = nn.Linear(latent_dim, latent_dim)
        self.logvar_c = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, c):

        c = self.c_processor(c)
        z = self.net(torch.cat([x, c], dim=1))
        return self.mu_c(z), self.logvar_c(z)




class Syn_Generator(nn.Module):
    def __init__(self, data_dim, c_dim, latent_dim, hidden=[16]):
        super(Syn_Generator, self).__init__()

        self.c_processor = C_Processor(c_dim, latent_dim, hidden)
        self.net = DenseNN(input_dim=2 * latent_dim, hidden_dims=hidden, param_dims=[data_dim], nonlinearity=nn.ReLU())
        self.nc_net = DenseNN(input_dim=latent_dim, hidden_dims=hidden, param_dims=[data_dim], nonlinearity=nn.ReLU())

    def forward(self, z, c):

        c = self.c_processor(c)
        return self.net(torch.cat([z, c], dim=1))