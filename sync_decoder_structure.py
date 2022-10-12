from utils import *
from pyro.nn import DenseNN

device = select_device()

class SampleBox(nn.Module):
    def __init__(self, d, r, t, kappa, mix_param, comp_param, device, seed=123):
        super(SampleBox, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.d = d
        self.r = r
        self.t = t
        self.kappa = kappa
        self.device = device

        self.XTransformer = nn.Linear(self.r, self.d)
        self.CGenerator = nn.Linear(self.t, self.t)

        mix = D.Categorical(probs=mix_param)
        comp = D.Independent(D.Normal(loc=comp_param[0], scale=comp_param[1]), 1)
        self.gmm = D.MixtureSameFamily(mix, comp)


    def generate_xnc(self, sample_size, c_tran, x_tran):
        # Mixed Gaussian
        with torch.no_grad():
            x = self.gmm.sample((sample_size,)) + torch.randn((sample_size, self.r), device=self.device)
            c = c_tran(self.CGenerator(x[:, :self.t]))
            x = x_tran(self.XTransformer(x))
        return x, c

class Encoder(nn.Module):
    def __init__(self, d, kappa, t, hidden=[16]):
        super(Encoder, self).__init__()

        self.c_processor = DenseNN(t, hidden, param_dims=[kappa])
        self.net = DenseNN(kappa + d, hidden_dims=hidden, param_dims=[kappa], nonlinearity=nn.PReLU())

        self.mu = nn.Linear(kappa, kappa)
        self.logvar = nn.Linear(kappa, kappa)

    def forward(self, x, c):
        c = self.c_processor(c)
        z = self.net(torch.cat([x, c], dim=1))
        return self.mu(z), self.logvar(z)

class Decoder(nn.Module):
    def __init__(self, d, kappa, t, hidden=[16]):
        super(Decoder, self).__init__()

        self.c_processor = DenseNN(t, hidden, param_dims=[kappa], nonlinearity=nn.PReLU())
        self.net = DenseNN(2*kappa, hidden, param_dims=[d], nonlinearity=nn.PReLU())

    def forward(self, z, c):
        c = self.c_processor(c)
        x = self.net(torch.cat([z, c], dim=1))

        return x

class Prior(nn.Module):
    def __init__(self, kappa, t, hidden=[16]):
        super(Prior, self).__init__()

        self.c_processor = DenseNN(t, hidden, param_dims=[kappa], nonlinearity=nn.PReLU())
        self.prior_mu = nn.Linear(kappa, kappa)
        self.prior_logvar = nn.Linear(kappa, kappa)

    def forward(self, c):

        c = self.c_processor(c)
        return self.prior_mu(c), self.prior_logvar(c)

class Decoder_prior(nn.Module):
    def __init__(self, d, kappa, t, hidden=[16]):
        super(Decoder_prior, self).__init__()

        self.prior = DenseNN(t, hidden, param_dims=[kappa], nonlinearity=nn.PReLU())
        self.prior_mu = nn.Linear(kappa, kappa)
        self.prior_logvar = nn.Linear(kappa, kappa)

        self.c_processor = DenseNN(t, hidden, param_dims=[kappa], nonlinearity=nn.PReLU())
        self.net = DenseNN(2 * kappa, hidden, param_dims=[d], nonlinearity=nn.PReLU())

    def forward(self, z, c):

        cp = self.prior(c)
        mean, logvar = self.prior_mu(cp), self.prior_logvar(cp)
        z = mean + torch.exp(0.5 * logvar) * z

        c = self.c_processor(c)
        x = self.net(torch.cat([z, c], dim=1))

        return x



d = 20
kappa = 20
t = 5
coef = 0

nepoch = 20000
lr = 0.01
seed = 123

x_tran = lambda x: torch.sigmoid_(x)
c_tran = lambda x: torch.abs(x)**.5

mix_param = torch.rand((5,), device=device)
for r in [5, 10, 15]:
    save_path = f"synthetic/prior_in_decoder/False/{coef}_{d}_{kappa}_{r}_{t}"
    os.makedirs(save_path, exist_ok=True)

    comp_param = [torch.randn((5, r), device=device), torch.exp(torch.randn((5, r), device=device))]
    samplebox = SampleBox(d, r, t, kappa, mix_param, comp_param, device, seed).to(device)
    encoder = Encoder(d, kappa, t).to(device)
    generator = Decoder(d, kappa, t).to(device)
    prior = Prior(kappa, t).to(device)
    loggamma = nn.Parameter(coef * torch.ones(1, device=device))

    opt = torch.optim.Adam(list(encoder.parameters()) + list(generator.parameters()) + list(prior.parameters()) + [loggamma], lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch,
                                                    cycle_momentum=False)
    x, c = samplebox.generate_xnc(50000, c_tran, x_tran)
    torch.save(x, f"{save_path}/x.data")
    torch.save(c, f"{save_path}/c.data")

    t1 = time.time()
    for epoch in range(nepoch):
        muc, logvarc = prior(c)
        mean, logvar = encoder(x, c)
        postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
        kl = torch.sum(torch.exp(logvar - logvarc) + torch.square(mean - muc) / torch.exp(logvarc) - logvar + logvarc, dim=1) - kappa

        xhat = generator(postz, c)
        recon = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
        recon2 = d * loggamma + math.log(2 * math.pi) * d
        loss = torch.mean(recon + kl + recon2)

        gamma = torch.exp(loggamma)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 200 == 0:
            print(
                f"epoch = {epoch}, recon = {torch.mean(recon * gamma).item()}, kl = {torch.mean(kl).item()}, gamma = {gamma.data}")
    print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")

    torch.save(encoder, f"{save_path}/encoder.pth")
    torch.save(generator, f"{save_path}/generator.pth")
    torch.save(loggamma, f"{save_path}/loggamma.pth")
    torch.save(prior, f"{save_path}/prior.pth")


for r in [5, 10, 15]:
    save_path = f"synthetic/prior_in_decoder/True/{coef}_{d}_{kappa}_{r}_{t}"
    os.makedirs(save_path, exist_ok=True)

    comp_param = [torch.randn((5, r), device=device), torch.exp(torch.randn((5, r), device=device))]
    samplebox = SampleBox(d, r, t, kappa, mix_param, comp_param, device, seed).to(device)
    encoder = Encoder(d, kappa, t).to(device)
    generator = Decoder_prior(d, kappa, t).to(device)
    loggamma = nn.Parameter(coef * torch.ones(1, device=device))

    opt = torch.optim.Adam(list(encoder.parameters()) + list(generator.parameters()) + [loggamma], lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch,
                                                    cycle_momentum=False)
    x, c = samplebox.generate_xnc(50000, c_tran, x_tran)
    torch.save(x, f"{save_path}/x.data")
    torch.save(c, f"{save_path}/c.data")

    t1 = time.time()
    for epoch in range(nepoch):
        mean, logvar = encoder(x, c)
        postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
        kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - kappa

        xhat = generator(postz, c)
        recon = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
        recon2 = d * loggamma + math.log(2 * math.pi) * d
        loss = torch.mean(recon + kl + recon2)

        gamma = torch.exp(loggamma)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 200 == 0:
            print(
                f"epoch = {epoch}, recon = {torch.mean(recon * gamma).item()}, kl = {torch.mean(kl).item()}, gamma = {gamma.data}")
    print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")

    torch.save(encoder, f"{save_path}/encoder.pth")
    torch.save(generator, f"{save_path}/generator.pth")
    torch.save(loggamma, f"{save_path}/loggamma.pth")
