import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from train_merge import train_model
# 你的 UNet
from merge_model import UNet
from evaluate_merge import evaluate_model
# ===============================
# Run all models
# ===============================
for mode in ["M1", "M2", "M3"]:

    model = UNet(C=16, mode=mode).to(device)

    train_model(
        mode=mode,
        model=model,
        device=device,
        steps=20000
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {}

for mode in ["M1", "M2", "M3"]:

    model = UNet(C=16, mode=mode).to(device)

    ckpt = torch.load(f"./ckpt_{mode}.pth", map_location=device)
    model.load_state_dict(ckpt["model"])

    mse = evaluate_model(model, device)

    results[mode] = mse

print("\nFinal Results:")
for k, v in results.items():
    print(k, ":", v)