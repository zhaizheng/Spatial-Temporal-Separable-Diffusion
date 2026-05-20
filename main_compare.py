import os
import json
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from train_merge import train_model
from merge_model import UNet
from evaluate_merge import evaluate_model
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


def train_model(
    mode,
    model,
    device,
    steps,
    eval_steps,          # 👈 新增：要评估的 step list
    save_dir,
    evaluate_fn,
    batch_size = 256
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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


    results = {}

    for step in range(1, steps + 1):

        # ===== training step =====
        x0, _ = next(loader_iter)
        x0 = x0.to(device)

        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device)

        noise = torch.randn_like(x0)
        a_bar = alpha_bar[t].view(-1,1,1,1)
        xt = torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*noise

        pred = model(xt, t)

        loss = (noise - pred).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== evaluation trigger =====
        if step in eval_steps:
            print(f"\n>>> Eval at step {step}")

            ckpt_path = f"{save_dir}/model_step{step}.pth"
            torch.save({"model": model.state_dict()}, ckpt_path)

            mse = evaluate_fn(model, device)

            results[step] = float(mse)

            print(f"Step {step} | MSE: {mse}")

    return results


if __name__ == '__main__':

    channels_list = [8, 16, 24, 32, 40]
    modes = ["M1", "M2", "M3"]
    #modes = ["M1"]
    # 👇 关键：每个阶段都测
    eval_steps = [5000, 10000, 15000, 20000]

    os.makedirs("./models", exist_ok=True)

    all_results = {}

    for C in channels_list:
        for mode in modes:

            print(f"\n=== Training {mode}, C={C} ===")

            model = UNet(C=C, mode=mode).to(device)

            save_dir = f"./models/{mode}_C{C}"
            os.makedirs(save_dir, exist_ok=True)

            results = train_model(
                mode=mode,
                model=model,
                device=device,
                steps=max(eval_steps),
                eval_steps=eval_steps,
                save_dir=save_dir,
                evaluate_fn=evaluate_model
            )

            all_results[f"{mode}_C{C}"] = results

    # 保存所有结果
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nDone!")