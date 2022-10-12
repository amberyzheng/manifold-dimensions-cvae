from sync_models import *

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

device = torch.device("cuda:0")
nepoch = 100000
seed = 123
sample_size = 100000
d = 20
r = 10
t = 5
kappa = 20
bs = 1000
lr = 0.01
coefs = [-20, -10, -5, 0, 5, 10, 20]
coef = 0
ts = [2, 4, 6, 8, 10]
x_tran = lambda x: torch.sigmoid_(x)
c_tran = lambda x: x
mix_param = torch.tensor([0.2, 0.3, 0.5, 0.3, 0.2], device=device)


# Fixed prior
torch.manual_seed(seed)
comp_param = [torch.randn((5, r), device=device), torch.exp(torch.randn((5, r), device=device))]

samplebox = SampleBox(d, r, t, kappa, mix_param, comp_param, device, seed).to(device)
x, c = samplebox.generate_xnc(sample_size, c_tran, x_tran)

for coef in coefs:
    save_path = f"synthetic/std_prior_relu/{coef}_{d}_{kappa}_{r}_{t}"
    os.makedirs(save_path, exist_ok=True)


    encoder = Syn_Encoder(d, t, kappa).to(device)
    generator = Syn_Generator(d, t, kappa).to(device)
    loggamma = nn.Parameter(coef*torch.ones(1, device=device))
    opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma], lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch*(sample_size // bs), cycle_momentum=False)

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

        if (epoch+1) % 2000 == 0:
            print(f"epoch = {epoch}, loss = {loss}, recon = {torch.mean(recon*gamma).item()}, kl = {torch.mean(kl).item()}, gamma = {gamma.data}")
    print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")

    torch.save(encoder, f"{save_path}/encoder.pth")
    torch.save(generator, f"{save_path}/generator.pth")
    torch.save(loggamma, f"{save_path}/loggamma.pth")

    xt, ct = samplebox.generate_xnc(2000, c_tran, x_tran)
    mean, logvar = encoder(xt, ct)
    print(torch.mean(torch.exp(logvar), dim=0))


r = 10
coef = 0
torch.manual_seed(seed)
comp_param = [torch.randn((5, r), device=device), torch.exp(torch.randn((5, r), device=device))]

for t in [1, 3, 5, 7, 9]:
    samplebox = SampleBox(d, r, t, kappa, mix_param, comp_param, device, seed).to(device)
    torch.manual_seed(456)
    x, c = samplebox.generate_xnc(2000, c_tran, x_tran)

    save_path = f"synthetic/std_prior/{coef}_{d}_{kappa}_{r}_{t}"
    encoder = torch.load(f"{save_path}/encoder.pth").to(device)
    generator = torch.load(f"{save_path}/generator.pth").to(device)
    loggamma = torch.load(f"{save_path}/loggamma.pth").to(device)

    mean, logvar = encoder(x, c)
    postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
    kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - kappa

    xhat = generator(postz, c)
    recon = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
    recon2 = d * loggamma + math.log(2 * math.pi) * d
    loss = torch.mean(recon + kl + recon2)

    print(f"coef={coef}, nll={loss}, recon={recon.mean()*torch.exp(loggamma)}, kl={kl.mean()}")
    print(torch.exp(logvar).mean(dim=0))