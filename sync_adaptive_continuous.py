import torch
from pyro.nn import DenseNN
from utils import *

seed = 123
device = torch.device("cuda:1")
size = 1000

r = 12
d = 20
ts = [2, 4, 6, 8, 10]
tt = max(ts)
c_dim = max(ts)
name = f"sigmoid_sq_{ts}"
kappa = 90
atten = True
torch.manual_seed(seed)

class Syn_Encoder(nn.Module):
    def __init__(self, data_dim, c_dim, latent_dim, hidden=[16]):
        super(Syn_Encoder, self).__init__()

        self.c_processor = DenseNN(c_dim, hidden_dims=[32], param_dims=[c_dim])
        self.net = DenseNN(c_dim + data_dim, hidden_dims=hidden, param_dims=[latent_dim], nonlinearity=nn.PReLU())

        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, c):
        c = self.c_processor(c)
        z = self.net(torch.cat([x, c], dim=1))
        return self.mu(z), self.logvar(z)


class Syn_Generator(nn.Module):
    def __init__(self, data_dim, c_dim, latent_dim, atten=True, hidden=[16]):
        super(Syn_Generator, self).__init__()

        self.atten = atten
        self.c_dim = c_dim

        if atten:
            self.c_processor = DenseNN(c_dim, hidden_dims=[latent_dim+c_dim], param_dims=[c_dim, latent_dim], nonlinearity=nn.PReLU())
        else:
            self.c_processor = DenseNN(c_dim, hidden_dims=[32], param_dims=[c_dim], nonlinearity=nn.PReLU())
        self.net = DenseNN(input_dim=c_dim + latent_dim, hidden_dims=hidden, param_dims=[data_dim], nonlinearity=nn.PReLU())


    def forward(self, z, c):
        if self.atten:
            c1, weight = self.c_processor(c)
            self.weight = torch.sigmoid_(weight)
            z = self.weight * z
        else:
            c1 = self.c_processor(c)
        return self.net(torch.cat([z, c1], dim=1))


class Vanilla_Encoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden=[16]):
        super(Vanilla_Encoder, self).__init__()

        self.net = DenseNN(data_dim, hidden_dims=hidden, param_dims=[latent_dim], nonlinearity=nn.PReLU())

        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        z = self.net(x)
        return self.mu(z), self.logvar(z)


class Vanilla_Generator(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden=[16]):
        super(Vanilla_Generator, self).__init__()

        self.net = DenseNN(input_dim=latent_dim, hidden_dims=hidden, param_dims=[data_dim], nonlinearity=nn.PReLU())

    def forward(self, z):

        return self.net(z)


def mixgaussian_sampler(r, size):
    mix_param = torch.rand((5,), device=device)
    comp_param = [torch.randn((5, r), device=device), torch.exp(torch.randn((5, r), device=device))]
    mix = D.Categorical(probs=mix_param)
    comp = D.Independent(D.Normal(loc=comp_param[0], scale=comp_param[1]), 1)
    gmm = D.MixtureSameFamily(mix, comp)

    return gmm.sample((size,)) + torch.randn((size, r), device=device)


c_trans = []
for t in ts:
    c_trans.append(nn.Linear(t, tt).to(device))
x_tran = nn.Linear(r, d).to(device)

def gene_data(size, ts):
    xs = []
    cs = []

    with torch.no_grad():
        for t in ts:
            x = mixgaussian_sampler(r, size)
            cs.append(torch.cat([x[:, :t], torch.zeros((x.size(0), tt-t), device=device)], dim=1).float())
            xs.append(torch.sigmoid_(x_tran(x)).float())

        x = torch.concat(xs, dim=0)
        c = torch.concat(cs, dim=0)

    return x, c

coef = 0
lr = 0.001
nepoch = 50000
encoder_hidden = [64]
decoder_hidden = [64]
encoder = Syn_Encoder(d, c_dim, kappa, hidden=encoder_hidden).to(device)
generator = Syn_Generator(d, c_dim, kappa, atten=atten, hidden=decoder_hidden).to(device)
loggamma = nn.Parameter(coef*torch.ones(1, device=device))
opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma], lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch, cycle_momentum=False)

save_path = f"synthetic/continuous/{name}/{atten}/{coef}_{d}_{r}_{kappa}"
os.makedirs(save_path, exist_ok=True)

torch.manual_seed(seed)
x, c = gene_data(10000, ts)

t1 = time.time()
for epoch in range(nepoch):
    mean, logvar = encoder(x, c)
    postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
    kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - kappa

    xhat = generator(postz, c)
    recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
    recon2 = d * loggamma + math.log(2 * math.pi) * d
    loss = torch.mean(recon1 + recon2 + kl)

    gamma = torch.exp(loggamma)


    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()

    if epoch % 200 == 0:
        print(f"epoch = {epoch}, loss = {loss.item()}, kl = {torch.mean(kl)}, recon1 = {torch.mean(recon1*gamma)}, gamma= {gamma.data}")
print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")

torch.save(encoder, f"{save_path}/encoder.pth")
torch.save(generator, f"{save_path}/generator.pth")
torch.save(loggamma, f"{save_path}/loggamma.pth")


encoder = torch.load(f"{save_path}/encoder.pth")
generator = torch.load(f"{save_path}/generator.pth")


size = 1000
z = torch.randn((size*len(ts), kappa), device=device)
x, c = gene_data(size, ts)
mean, logvar = encoder(x, c)
postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - kappa

xhat = generator(postz, c)
recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
recon2 = d * loggamma + math.log(2 * math.pi) * d
loss = kl + recon1 + recon2

for i in range(len(ts)):
    print(ts[i])
    if atten:
        print((torch.mean(generator.weight[i*(size): (i+1)*(size)], dim=0)>0.1).sum())
    print(f"nll={loss[i*(size): (i+1)*(size)].mean()}, recon={(recon1*torch.exp(loggamma))[i*(size): (i+1)*(size)].mean()}, KL={kl[i*(size): (i+1)*(size)].mean()}")
    print((torch.exp(logvar).mean(dim=0)<0.01).sum())