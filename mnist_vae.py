from mnist_models import *

# Hyper parameters
case = "VAE"
date = datetime.date.today().strftime("%m%d")
os.makedirs(date, exist_ok=True)
ds = "mnist"
bs = 512
latent_dim = 32
lr = 0.002
base_channels = 5
nepoch = 400
loggamma_coef = 0
save_path = f"{date}/{ds}/{latent_dim}"
os.makedirs(save_path, exist_ok=True)


device = torch.device("cuda:0")
dat, test_dat, loader, test_label = load_data(bs, device, ds)
input_dim = dat.size(1) * dat.size(2)
c_dim = 0

encoder = Res_Encoder(latent_dim, c_dim, EncoderBlock, base_channels).to(device)
generator = Res_Decoder(latent_dim, c_dim, DecoderBlock, base_channels).to(device)
loggamma = nn.Parameter(loggamma_coef*torch.ones(1, device=device))
opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma], lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch * len(loader), cycle_momentum=False)

t1 = time.time()
for epoch in range(nepoch):
    for batch, data in enumerate(loader):
        x = data[0].to(device).view(-1, 784)
        mean, logvar = encoder(x)
        postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
        kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - latent_dim

        xhat = generator(postz)
        recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
        recon2 = input_dim * loggamma + math.log(2 * math.pi) * input_dim
        loss = torch.mean(recon1 + recon2 + kl, dim=0)
        gamma = torch.exp(loggamma)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    print(f"epoch = {epoch}, loss = {loss.item()}, kl = {torch.mean(kl)}, recon1 = {torch.mean(recon1*gamma)}, recon2 = {torch.mean(recon2)}, gamma= {gamma.data}")
print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")


## Save models
torch.save(encoder, f"{save_path}/encoder.pth")
torch.save(generator, f"{save_path}/generator.pth")
torch.save(loggamma, f"{save_path}/loggamma.pth")


encoder = torch.load(f"{save_path}/encoder.pth")
generator = torch.load(f"{save_path}/generator.pth")
loggamma = torch.load(f"{save_path}/loggamma.pth")

# View the latent dimensions pattern via encoder
x = test_dat[:100].view(100, -1)

mean, logvar = encoder(x)
postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - latent_dim

xhat = generator(postz)
recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
recon2 = input_dim * loggamma + math.log(2 * math.pi) * input_dim
loss = torch.mean(recon1 + recon2 + kl, dim=0)
gamma = torch.exp(loggamma)

print(f"nll={loss}, recon={recon1.mean()*gamma}, KL={kl.mean()}")
print(torch.exp(logvar).mean(dim=0))