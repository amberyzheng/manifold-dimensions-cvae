import torch

from mnist_models import *

date = datetime.date.today().strftime("%m%d")
ds = "fmnist"
name = "adaptive"
device = torch.device("cuda:2")
bs = 512
latent_dim = 32
lr = 0.002
base_channels = 32
nepoch = 400
coef = 0
save_path = f"{date}/{name}/{ds}"

dat, test_dat, loader, test_label = load_data(bs, device, ds)
input_dim = dat.size(1) * dat.size(2)
c_dim = 10

encoder = Res_Encoder(latent_dim, c_dim, EncoderBlock, base_channels).to(device)
generator = Res_Decoder(latent_dim, c_dim, DecoderBlock, base_channels).to(device)
loggamma = nn.Parameter(coef*torch.ones(1, device=device))
opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma], lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch * len(loader), cycle_momentum=False)


kls = []
t1 = time.time()
for epoch in range(nepoch):
    for batch, (x, label) in enumerate(loader):
        x, label = x.view(-1, input_dim).to(device), label.to(device)
        c = torch.zeros((x.size(0), c_dim), device=device).scatter_(1, label.view(-1, 1), 1)
        mean, logvar = encoder(x, c)
        postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
        kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - latent_dim
        kls.append(torch.mean(kl).item())

        xhat = generator(postz, c)
        recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
        recon2 = input_dim * loggamma + math.log(2 * math.pi) * input_dim
        loss = torch.mean(recon1 + recon2 + kl, dim=0)
        gamma = torch.exp(loggamma)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    print(f"epoch = {epoch}, loss = {loss.item()}, kl = {torch.mean(kl)}, recon1 = {torch.mean(recon1 * gamma)}, recon2 = {torch.mean(recon2)}, gamma= {gamma.data}")
print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")

## Save models
os.makedirs(save_path, exist_ok=True)
torch.save(encoder, f"{save_path}/encoder.pth")
torch.save(generator, f"{save_path}/generator.pth")
torch.save(loggamma, f"{save_path}/loggamma.pth")
np.save(f"{save_path}/kl.npy", np.array(kls))