from pyro.nn import DenseNN
from utils import *

class EncoderBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels * EncoderBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * EncoderBlock.expansion)
        )

        #shortcut
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != EncoderBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * EncoderBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * EncoderBlock.expansion)
            )
        else:
            self.shortcut = nn.Identity()

        self.activate = nn.PReLU()

    def forward(self, x):
        return self.activate(self.residual_function(x) + self.shortcut(x))

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        #conv transpose
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.convt(self.residual_function(x) + x)

class Res_Encoder(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels):
        super(Res_Encoder, self).__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[32], param_dims=[latent_dim, latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x.view(-1, 1, 28, 28))
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            mean, logvar = self.z(x)
        return mean, logvar

class Res_Decoder(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels=32, atten=False):
        super(Res_Decoder, self).__init__()
        self.atten = atten
        self.in_channels = in_channels

        if c_dim > 0:
            if atten:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2, latent_dim], nonlinearity=nn.PReLU())
            else:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())

        self.upsample = nn.Linear(in_features=latent_dim+c_dim//2, out_features=7*7*in_channels)

        self.net = nn.Sequential(
            block(in_channels, in_channels),
            block(in_channels, in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Flatten()
        )

    def forward(self, z, c=None):
        if c != None:
            if self.atten:
                c, weight = self.cprocesser(c)
                self.weight = torch.sigmoid_(weight)
                z = z * self.weight
            else:
                c = self.cprocesser(c)
            z = self.upsample(torch.cat([z, c], dim=1)).relu_().view(-1, self.in_channels, 7, 7)
        else:
            z = self.upsample(z).relu_().view(-1, self.in_channels, 7, 7)
        x = self.net(z)
        return torch.sigmoid_(x)
