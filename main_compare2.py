import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
from itertools import cycle

from merge_model import UNet
from evaluate_merge import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)
    

def orthogonal_loss(f):
        # f: (B, C, H, W)
    B, C, H, W = f.shape

    # 👉 展平空间维度
    f_flat = f.view(B, C, -1)   # (B, C, HW)

    # 👉 Gram matrix
    gram = torch.bmm(f_flat, f_flat.transpose(1, 2))  # (B, C, C)

    # 👉 归一化（关键！避免 scale 影响）
    gram = gram / (H * W)

    # 👉 Identity
    I = torch.eye(C, device=f.device).unsqueeze(0)  # (1, C, C)

    # 👉 orth loss
    return ((gram - I) ** 2).mean()

# =========================================================
# 🔥 TRAIN FUNCTION (EMA + Resume)
# =========================================================
def train_model(
    mode,
    model,
    device,
    steps,
    eval_steps,
    save_dir,
    evaluate_fn,
    C,
    batch_size=256
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ===============================
    # 🔥 EMA model
    # ===============================
    ema_model = UNet(C=C, mode=mode).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = 0.999

    # ===============================
    # 🔥 Resume from checkpoint
    # ===============================
    latest_ckpt = os.path.join(save_dir, "latest.pth")
    start_step = 1

    if os.path.exists(latest_ckpt):
        print(f"🔄 Resume from {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, map_location=device)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1

        if "ema" in ckpt:
            ema_model.load_state_dict(ckpt["ema"])

        print(f"✅ Resumed at step {start_step}")

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
    #beta = cosine_beta_schedule(T).to(device)
    beta = linear_beta_schedule(T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    results = {}

    # ===============================
    # training loop
    # ===============================
    for step in range(start_step, steps + 1):

        x0, _ = next(loader_iter)
        x0 = x0.to(device)

        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device)

        noise = torch.randn_like(x0)
        a_bar = alpha_bar[t].view(-1,1,1,1)
        xt = torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*noise

        pred, f = model(xt, t)
        if mode == "M2orth":
            loss = (noise - pred).pow(2).mean() + 0.1 * orthogonal_loss(f)

        elif mode == "M2orth_smooth":
            delta_t = torch.randint(-5, 6, (B,), device=device)

            t2 = (t + delta_t).clamp(0, T - 1)

            t_emb1 = model.time_embed(t)
            t_emb2 = model.time_embed(t2)

            w1 = model.time_net(t_emb1)
            w2 = model.time_net(t_emb2)

            smooth_loss = (w1 - w2).pow(2).mean()
            loss = (noise - pred).pow(2).mean() + 0.1 * orthogonal_loss(f)+0.1 * smooth_loss
        else:
            loss = (noise - pred).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===============================
        # 🔥 EMA update
        # ===============================
        with torch.no_grad():
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        # ===============================
        # evaluation + save
        # ===============================
        if step in eval_steps:

            print(f"\n>>> Eval at step {step}")

            # 🔥 save latest (for resume)
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, latest_ckpt)

            # 🔥 save historical checkpoint
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict()
            }, f"{save_dir}/model_step{step}.pth")

            # 🔥 evaluate EMA
            mse = evaluate_fn(ema_model, device)

            results[step] = float(mse)

            print(f"Step {step} | MSE: {mse:.6f}")

    return results


# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':

    channels_list = [8, 16, 24]#, 32, 40]
    #channels_list = [2,4,6]#[2, 4, 6, 8, 10]
    modes = ["M2orth_smooth"]#["M1","M2","M2orth","M2orth_smooth","M3"]#modes = ["M1","M2","M2orth","M3"]

    eval_steps = [5000, 10000,15000, 20000, 25000, 30000, 35000, 40000] #, 15000, 20000, 25000, 30000, 35000, 40000, 50000]#[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]

    os.makedirs("./models", exist_ok=True)

    all_results = {}

    for C in channels_list:
        for mode in modes: 

            print(f"\n=== Training {mode}, C={C} ===")

            model= UNet(C=C, mode=mode).to(device)
            if mode == "M2orth":
                modezz = "M2"
                save_dir = f"./models/{modezz}_C{C}_orth"
            elif mode == "M2orth_smooth":
                modezz = "M2"
                save_dir = f"./models/{modezz}_C{C}_orth_smooth"
            else:
                save_dir = f"./models/{mode}_C{C}"
            os.makedirs(save_dir, exist_ok=True)

            results = train_model(
                mode=mode,
                model=model,
                device=device,
                steps=max(eval_steps),
                eval_steps=eval_steps,
                save_dir=save_dir,
                evaluate_fn=evaluate_model,
                C=C   # 🔥 必须传
            )

            all_results[f"{mode}_C{C}"] = results

    # ===============================
    # save results
    # ===============================
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n✅ Done!")