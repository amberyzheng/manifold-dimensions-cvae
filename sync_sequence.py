from statsmodels.tsa.arima_process import arma_generate_sample
from utils import *

seed = 123
device = torch.device("cuda:2")
size = 20000
bs = 2000
nepoch = 10000
T = 6
r = 10
d = 20
kappa = 20
coef = 0
lr = 0.01
hidden_size = 32
sharing = True
torch.manual_seed(seed)

def rolling_window(x, window_size, step_size=1):
    return x.unfold(0, window_size, step_size).permute(0, 2, 1)


def sampler(size, r):
    ar_coefs = np.array([1, -.5, .25, -.4])
    ma_coefs = np.array([1, .65, .35])
    np.random.seed(123)

    data = []
    for i in range(r):
        data.append(torch.tensor(arma_generate_sample(ar_coefs, ma_coefs, nsample=size, scale=0.5).reshape(-1, 1), device=device))

    return torch.cat(data, dim=1).float()

torch.manual_seed(seed)
layer = nn.Linear(r, d).to(device)
def gene_data(size):
    with torch.no_grad():
        data = layer(sampler(size, r))
    return rolling_window(data, T)


class Seq_Encoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_size):
        super(Seq_Encoder, self).__init__()

        self.net = nn.LSTM(data_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        out = self.net(x)[0][: , -1]
        return self.mu(out), self.logvar(out)


class Seq_Decoder(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_size):
        super(Seq_Decoder, self).__init__()

        self.c_transform = nn.LSTM(data_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.output = nn.Linear(hidden_size+latent_dim, data_dim)

    def forward(self, z, c=None):
        c = self.c_transform(c)[0][:, -1]
        out = self.output(torch.cat([z, c], dim=1))
        return out

torch.manual_seed(seed)
data = gene_data(size)
X = data[:, -1, :].view(-1, 1, d)
C = data[:, :-1, :]


encoder = Seq_Encoder(d, kappa, hidden_size).to(device)
if sharing:
    prior = encoder
else:
    prior = Seq_Encoder(d, kappa, hidden_size).to(device)
generator = Seq_Decoder(d, kappa, hidden_size).to(device)
loggamma = nn.Parameter(coef*torch.ones(1, device=device))
opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma]+list(prior.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch*(size//bs), cycle_momentum=False)

t1 = time.time()
for epoch in range(nepoch):
    for i in range(size // bs):
        x, c = X[i*bs: (i+1)*bs], C[i*bs: (i+1)*bs]
        mean, logvar = encoder(torch.cat([c, x], dim=1))
        meanp, logvarp = prior(c)
        kl = torch.sum(torch.exp(logvar - logvarp) + torch.square(mean - meanp) / torch.exp(logvarp) - logvar + logvarp, dim=1) - kappa

        postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)

        xhat = generator(postz, c)
        recon1 = torch.sum(torch.square(x.squeeze() - xhat), dim=1) / torch.exp(loggamma)
        recon2 = d * loggamma + math.log(2 * math.pi) * d
        loss = torch.mean(recon1 + recon2 + kl)

        gamma = torch.exp(loggamma)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    if epoch % 50 == 0:
        print(
            f"epoch = {epoch}, loss = {loss.item()}, kl = {torch.mean(kl)}, recon1 = {torch.mean(recon1 * gamma)}, gamma= {gamma.data}")
print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")


save_path = f"sequence/{sharing}/{size}_{nepoch}_T{T}"
os.makedirs(save_path, exist_ok=True)
torch.save(encoder, f"{save_path}/encoder.pth")
torch.save(generator, f"{save_path}/generator.pth")
torch.save(prior, f"{save_path}/prior.pth")
torch.save(loggamma, f"{save_path}/loggamma.pth")

data = gene_data(1000)
x = data[:, -1, :].view(-1, 1, d)
c = data[:, :-1, :]

mean, logvar = encoder(torch.cat([c, x], dim=1))
meanp, logvarp = prior(c)
kl = torch.sum(torch.exp(logvar - logvarp) + torch.square(mean - meanp) / torch.exp(logvarp) - logvar + logvarp, dim=1) - kappa

postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)

xhat = generator(postz, c)
recon1 = torch.sum(torch.square(x.squeeze() - xhat), dim=1) / torch.exp(loggamma)
recon2 = d * loggamma + math.log(2 * math.pi) * d
loss = torch.mean(recon1 + recon2 + kl)

gamma = torch.exp(loggamma)
print(f"nll={loss}, recon={(recon1*gamma).mean()}, kl={kl.mean()}")