import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from itertools import cycle

# ===============================
# Diffusion schedule
# ===============================
T = 1000

def cosine_beta_schedule(T, s=0.008):
    x = torch.linspace(0, T, T+1)
    alphas_bar = torch.cos(((x/T)+s)/(1+s)*torch.pi*0.5)**2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-4, 0.999)

# ===============================
# Train function
# ===============================
def train_model(mode, model, device, steps=30000, batch_size=256):

    print(f"\n🚀 Training {mode}")

    # ===============================
    # dataset
    # ===============================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader_iter = cycle(loader)

    # ===============================
    # diffusion
    # ===============================
    beta = cosine_beta_schedule(T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # ===============================
    # optimizer
    # ===============================
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ===============================
    # checkpoint (only ONE file)
    # ===============================
    ckpt_path = f"./ckpt_{mode}.pth"

    start_step = 0

    # 🔥 自动恢复
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]

        print(f"✅ Resume from step {start_step}")
    else:
        print("🆕 Training from scratch")

    # ===============================
    # training loop
    # ===============================
    for step in range(start_step, steps):

        x0, _ = next(loader_iter)
        x0 = x0.to(device)

        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device)

        noise = torch.randn_like(x0)
        a_bar = alpha_bar[t].view(-1,1,1,1)
        xt = torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*noise

        pred = model(xt, t)

        #loss = (noise - pred).pow(2).mean()
        alpha_bar_t = a_bar.view(-1)
        snr = alpha_bar_t / (1 - alpha_bar_t + 1e-5)
        err = (noise - pred).pow(2).mean(dim=(1,2,3)) 
        loss = (snr * err).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===============================
        # log
        # ===============================
        if step % 100 == 0:
            print(f"{mode} | step {step} | loss {loss.item():.4f}")

        # ===============================
        # 🔥 overwrite save (only ONE file)
        # ===============================
        if step % 1000 == 0 and step > start_step:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, ckpt_path)

            print(f"💾 Saved (overwrite): {ckpt_path}")

    # ===============================
    # final save
    # ===============================
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": steps
    }, ckpt_path)

    print(f"✅ {mode} training done")