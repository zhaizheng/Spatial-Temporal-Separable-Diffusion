import os
import re
import torch
from merge_model import UNet
from evaluate_merge import evaluate_model


import os
import torch
import matplotlib.pyplot as plt



def plot_time_net_lines(model, C, device, save_path, title):

    model.eval()

    ts = torch.linspace(0, 1000, 300).to(device)  # finer resolution
    t_emb = model.time_embed(ts)

    try:
        with torch.no_grad():
            w = model.time_net(t_emb)  # (T, C)
            w = w.detach().cpu()  # (T, C)
    except Exception as e:
        print(f"⚠️ Failed {title}: {e}")
        return
    num_channels = C 
    # ============================
    # select channels
    # ============================
    idx = torch.linspace(0, C - 1, num_channels).long()

    plt.figure(figsize=(6, 4))

    for i in idx:
        # 改成散点
        plt.scatter(ts.cpu(), w[:, i], s=10, label=f"c{i.item()}")

    plt.xlabel("time t")
    plt.ylabel("w(t)")
    plt.title(title)
    plt.legend(fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_net_norm(model, device, save_path, title):
    import os
    import torch
    import matplotlib.pyplot as plt

    model.eval()

    ts = torch.linspace(0, 1000, 300).to(device)
    t_emb = model.time_embed(ts)

    try:
        with torch.no_grad():
            w = model.time_net(t_emb)      # shape: (T, C)
            w = w.detach().cpu()

            # ============================
            # compute ||w(t)||
            # ============================
            w_norm = torch.norm(w, dim=1)  # shape: (T,)
    except Exception as e:
        print(f"⚠️ Failed {title}: {e}")
        return

    # ============================
    # plot
    # ============================
    plt.figure(figsize=(6, 4))

    # 改成散点
    plt.scatter(ts.cpu(), w_norm, s=12)

    plt.xlabel("time t")
    plt.ylabel(r"$\|w(t)\|$")
    plt.title(title)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
# =========================================================
# 🔍 找到所有 ckpt
# =========================================================
def find_all_ckpts(folder):

    if not os.path.exists(folder):
        return []

    pattern = r"model_step(\d+)\.pth"
    ckpts = []

    for f in os.listdir(folder):
        match = re.search(pattern, f)
        if match:
            step = int(match.group(1))
            ckpts.append((step, f))

    ckpts.sort(key=lambda x: x[0])  # 按 step 排序

    return [(step, os.path.join(folder, f)) for step, f in ckpts]


# =========================================================
# Load model
# =========================================================
def load_model(model_class, ckpt_path, device, C, mode):

    try:
        model = model_class(C=C, mode=mode).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)

        if "ema" in ckpt:
            state_dict = ckpt["ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    except Exception as e:
        print(f"❌ Failed: {ckpt_path}")
        print("Error:", e)
        return None


# =========================================================
# 🔥 加载所有 step 的模型
# =========================================================
def load_all_models(model_class, root="./models", device="cuda"):

    model_dict = {}

    for folder in os.listdir(root):

        # 解析 M2_C32 / M2_C32orth
        match = re.match(r"(M\d+)_C(\d+)", folder)
        if not match:
            print(f"Skip: {folder}")
            continue

        mode = match.group(1)
        C = int(match.group(2))

        full_path = os.path.join(root, folder)

        if not os.path.isdir(full_path):
            continue

        ckpt_list = find_all_ckpts(full_path)

        if len(ckpt_list) == 0:
            print(f"❌ No ckpt in {folder}")
            continue

        print(f"\n📂 {folder} | {len(ckpt_list)} ckpts")

        for step, ckpt_path in ckpt_list:

            model = load_model(
                model_class,
                ckpt_path,
                device,
                C=C,
                mode=mode
            )

            if model is None:
                continue

            key = f"{folder}_step{step}"
            model_dict[key] = model

    return model_dict


# =========================================================
# 🔥 Evaluate all
# =========================================================

def evaluate_all(models, device):

    results = []

    print("\n📊 Evaluating ALL checkpoints...")

    for name, model in models.items():

        print(f"\nEvaluate: {name}")

        # ============================
        # parse info
        # ============================
        step = int(re.search(r"step(\d+)", name).group(1))
        C = int(re.search(r"_C(\d+)", name).group(1))
        mode = re.search(r"(M\d+)", name).group(1)

        # ============================
        # MSE evaluation
        # ============================
        mse = evaluate_model(model, device=device)

        results.append({
            "name": name,
            "mode": mode,
            "C": C,
            "step": step,
            "mse": mse
        })

        # ============================
        # 🔥 TimeNet visualization
        # ============================
        fig_path = f"figures/{name}_timenet.png"

        plot_time_net_lines(
            model=model,
            C=C,
            device=device,
            save_path=fig_path,
            title=name
        )

        fig_path2 = f"figures/{name}_timenet_norm.png"
        plot_time_net_norm(
            model=model,
            device=device,
            save_path=fig_path2,
            title=name
        )

        print(f"📊 Saved figure: {fig_path}")

        # ============================
        # cleanup
        # ============================
        del model
        torch.cuda.empty_cache()

    return results
'''
def evaluate_all(models, device):

    results = []

    print("\n📊 Evaluating ALL checkpoints...")

    for name, model in models.items():

        print(f"\nEvaluate: {name}")

        # 🔥 解析信息
        step = int(re.search(r"step(\d+)", name).group(1))
        C = int(re.search(r"_C(\d+)", name).group(1))
        mode = re.search(r"(M\d+)", name).group(1)

        mse = evaluate_model(model, device=device)

        results.append({
            "name": name,
            "mode": mode,
            "C": C,
            "step": step,
            "mse": mse
        })

        # 🔥 防止显存爆
        del model
        torch.cuda.empty_cache()

    return results
'''

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(ROOT, "models")

    print("📂 MODEL_DIR:", MODEL_DIR)

    models = load_all_models(UNet, MODEL_DIR, device)

    if len(models) == 0:
        print("❌ No models loaded")
        exit()

    results = evaluate_all(models, device)

    # =====================================================
    # 打印结果
    # =====================================================
    print("\n📈 Final Results:")
    for r in results:
        print(f"{r['name']} | MSE={r['mse']:.6f}")

    # =====================================================
    # （可选）保存结果
    # =====================================================
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)