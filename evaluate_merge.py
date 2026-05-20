import torch
import torchvision
import torchvision.transforms as transforms

# ===============================
# Evaluate function
# ===============================
@torch.no_grad()
def evaluate_model(model, device, batch_size=256):

    model.eval()

    # 🔥 固定随机种子（核心）
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # ===============================
    # dataset (train set 全部数据)
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

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False   # 🔥 必须 False
    )

    # ===============================
    # diffusion schedule
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

    #beta = cosine_beta_schedule(T).to(device)
    beta = linear_beta_schedule(T).to(device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # ===============================
    # evaluation
    # ===============================
    total_loss = 0
    total_samples = 0

    for x0, _ in loader:

        x0 = x0.to(device)
        B = x0.size(0)

        # 🔥 固定随机性（每个 batch 仍 deterministic）
        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(x0)

        a_bar = alpha_bar[t].view(-1,1,1,1)
        xt = torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*noise

        pred, _ = model(xt, t)

        loss = (noise - pred).pow(2).mean()

        total_loss += loss.item() * B
        total_samples += B

    mse = total_loss / total_samples

    print(f"📊 MSE: {mse:.6f}")
    return mse