import os
import re
import torch
import torchvision.utils as vutils
from merge_model import UNet

# =========================================================
# 🔍 自动找到最大 step 的 ckpt
# =========================================================
def find_latest_ckpt(folder):

    if not os.path.exists(folder):
        return None

    files = os.listdir(folder)

    pattern = r"model_step(\d+)\.pth"

    max_step = -1
    best_file = None

    print(f"\n📂 Checking {folder}")
    print("Files:", files)

    for f in files:
        match = re.search(pattern, f)   # 🔥 改这里
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                best_file = f

    if best_file is None:
        print("❌ No valid ckpt found")
        return None

    print(f"✅ Found ckpt: {best_file}")

    return os.path.join(folder, best_file)


# =========================================================
# Load model
# =========================================================
def load_model(model_class, ckpt_path, device, C, mode):

    print(f"\n🚀 Loading {mode}, C={C}")

    try:
        model = model_class(C=C, mode=mode).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)

        #state_dict = ckpt["model"] if "model" in ckpt else ckpt
        state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print("⚠️ Missing keys:", len(missing))
        print("⚠️ Unexpected keys:", len(unexpected))

        model.eval()

        print(f"✅ Loaded: {ckpt_path}")
        return model

    except Exception as e:
        print(f"❌ FAILED: {ckpt_path}")
        print("Error:", e)
        return None


# =========================================================
# 自动扫描 /models
# =========================================================
'''
def load_all_models(model_class, root="./models", device="cuda"):

    model_dict = {}

    folders = os.listdir(root)

    for folder in folders:
        # 解析 M1_16
        try:
            mode, C = folder.split("_")
            C = int(C.replace("C", ""))
        except:
            continue

        full_path = os.path.join(root, folder)

        ckpt_path = find_latest_ckpt(full_path)

        if ckpt_path is None:
            print(f"❌ No ckpt in {folder}")
            continue

        model = load_model(
            model_class,
            ckpt_path,
            device,
            C=C,
            mode=mode
        )
        if model is None:
            continue   # 🔥 一定要加
        key = f"{mode}_C{C}"
        model_dict[key] = model

    return model_dict
'''
import os
import re

def load_all_models(model_class, root="./models", device="cuda"):

    model_dict = {}
    folders = os.listdir(root)

    for folder in folders:
        # 🔥 robust parsing
        match = re.match(r"(M\d+)_C(\d+)", folder)
        if not match:
            print(f"Skip invalid folder: {folder}")
            continue

        mode = match.group(1)
        C = int(match.group(2))

        full_path = os.path.join(root, folder)
        ckpt_path = find_latest_ckpt(full_path)

        if ckpt_path is None:
            print(f"❌ No ckpt in {folder}")
            continue

        model = load_model(
            model_class,
            ckpt_path,
            device,
            C=C,
            mode=mode
        )
        if model is None:
            continue

        key = f"{mode}_C{C}"
        model_dict[key] = model

    return model_dict

# =========================================================
# DDIM Sampler
# =========================================================
@torch.no_grad()
def ddim_sample(model, device, shape=(64,1,28,28), steps=50, eta=0.0):

    model.eval()

    B = shape[0]
    T = 1000

    x = torch.linspace(0, T, T+1)
    alphas_bar = torch.cos(((x/T)+0.008)/(1+0.008)*torch.pi*0.5)**2
    alphas_bar = alphas_bar / alphas_bar[0]

    beta = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    beta = beta.clamp(1e-4, 0.999)

    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0).to(device)

    t_seq = torch.linspace(T-1, 0, steps, device=device).long()

    x = torch.randn(shape, device=device)

    for i in range(steps-1):

        t = t_seq[i]
        t_next = t_seq[i+1]

        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        eps,_  = model(x, t_batch)

        a_bar_t = alpha_bar[t]
        a_bar_next = alpha_bar[t_next]

        x0 = (x - torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(a_bar_t)

        sigma = (
            eta
            * torch.sqrt((1 - a_bar_next) / (1 - a_bar_t))
            * torch.sqrt(1 - a_bar_t / a_bar_next)
        )

        noise = torch.randn_like(x) if eta > 0 else 0.0

        x = (
            torch.sqrt(a_bar_next) * x0
            + torch.sqrt(1 - a_bar_next - sigma**2) * eps
            + sigma * noise
        )

    return x


# =========================================================
# 分组
# =========================================================
def group_by_C(models):

    grouped = {}

    for name, model in models.items():
        mode, C = name.split("_C")
        C = int(C)

        if C not in grouped:
            grouped[C] = {}

        grouped[C][mode] = model

    return grouped


# =========================================================
# 对比
# =========================================================
def compare_sampling_steps(model_dict, device, C, steps_list=[10,20,50,100]):

    os.makedirs("samples", exist_ok=True)

    for mode, model in model_dict.items():

        for steps in steps_list:

            samples = ddim_sample(
                model,
                device,
                steps=steps
            )

            samples = (samples.clamp(-1,1) + 1) / 2

            save_path = f"samples/{mode}_C{C}_steps{steps}.png"

            vutils.save_image(samples, save_path, nrow=8)

            print(f"✅ {save_path}")

            save_path2 = f"samples/{mode}_C{C}_steps{steps}_channels_t500.png"

            visualize_channels(model, device, save_path2, t_value=500, sample_idx=0):

            save_path3 = f"samples/{mode}_C{C}_steps{steps}_channels_t100.png"

            visualize_channels(model, device, save_path3, t_value=100, sample_idx=0):

def visualize_channels(model, device, save_path, t_value=500, sample_idx=0):

    import os
    import math
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    model.eval()

    # ==========================================
    # dataset
    # ==========================================
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

    x0, _ = dataset[sample_idx]

    x0 = x0.unsqueeze(0).to(device)   # (1,1,H,W)

    # ==========================================
    # diffusion schedule
    # ==========================================
    T = 1000

    beta = torch.linspace(1e-4, 0.02, T).to(device)

    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # ==========================================
    # generate x_t
    # ==========================================
    t = torch.tensor([t_value], device=device)

    noise = torch.randn_like(x0)

    a_bar = alpha_bar[t].view(-1,1,1,1)

    xt = (
        torch.sqrt(a_bar) * x0
        + torch.sqrt(1 - a_bar) * noise
    )

    # ==========================================
    # forward
    # ==========================================
    with torch.no_grad():

        pred, f = model(xt, t)

        # f shape:
        # (1, C, H, W)

        f = f.squeeze(0).cpu()

    C = f.shape[0]

    # ==========================================
    # plot all channels
    # ==========================================
    cols = 4
    rows = math.ceil(C / cols)

    plt.figure(figsize=(3*cols, 3*rows))

    for i in range(C):

        plt.subplot(rows, cols, i+1)

        plt.imshow(f[i], cmap='gray')

        plt.title(f"Channel {i}")

        plt.axis("off")

    plt.suptitle(f"Feature channels at t={t_value}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


# =========================================================
# MAIN
# =========================================================
def run(model_class, device):

    # ===============================
    # 路径修复（关键）
    # ===============================
    ROOT = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(ROOT, "models")

    print("📂 MODEL_DIR =", MODEL_DIR)

    if not os.path.exists(MODEL_DIR):
        print("❌ models folder NOT found!")
        return

    print("📁 Folders:", os.listdir(MODEL_DIR))

    # ===============================
    # 加载模型
    # ===============================
    models = load_all_models(model_class, MODEL_DIR, device)
    print("📊 Loaded models:", models.keys())
    print("📊 Loaded models:", models.keys())

    if len(models) == 0:
        print("❌ No models loaded. STOP.")
        return

    # ===============================
    # 分组
    # ===============================
    grouped = group_by_C(models)

    print("📦 Grouped keys:", grouped.keys())

    # ===============================
    # sampling（先跑最小）
    # ===============================
    for C, model_dict in grouped.items():

        print(f"\n🔥 Compare C={C}")
        print("Modes:", model_dict.keys())

        # 🔥 先只跑一个，避免卡死
        compare_sampling_steps(
            model_dict,
            device,
            C,
            steps_list=[500]   # 👈 先最小测试
        )

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run(UNet, device)