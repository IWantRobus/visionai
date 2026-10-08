"""
VISIONAI CORE — a DDPM (denoising diffusion probabilistic model) trained from scratch.
Architecture: small U-Net with sinusoidal timestep embeddings.
Dataset: MNIST handwritten digits (60,000 images).
This is the same class of model behind pro image generators — scaled to one CPU core.

Written for Tioluwalope Adesuyan's VISIONAI project. MIT license.
"""
import math, os, struct, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
DEV = "cpu"

# ---------------- Dataset (MNIST, parsed from raw idx files) ----------------
def load_mnist(path="../data"):
    with open(f"{path}/train-images-idx3-ubyte", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = torch.frombuffer(f.read(), dtype=torch.uint8).float()
    imgs = data.view(n, 1, rows, cols) / 127.5 - 1.0   # normalize to [-1, 1]
    with open(f"{path}/train-labels-idx1-ubyte", "rb") as f:
        f.read(8)
        labels = torch.frombuffer(f.read(), dtype=torch.uint8).long()
    print(f"Loaded {n} images {rows}x{cols}")
    return imgs, labels

# ---------------- Diffusion schedule (cosine) ----------------
def cosine_alphas(T=500, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    ab_t = alphas_bar[1:]                                   # alpha_bar at t=1..T
    a_t = ab_t / torch.cat([alphas_bar[0:1], ab_t[:-1]])    # alpha_t
    return {
        "alpha_bar": ab_t.float().view(-1, 1, 1, 1),
        "alpha": a_t.float().view(-1, 1, 1, 1),
        "beta": (1 - a_t).float().view(-1, 1, 1, 1),
    }

# ---------------- U-Net ----------------
class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_c):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.emb = nn.Linear(emb_c, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    """Small diffusion U-Net: 28x28 -> 14x14 -> 7x7."""
    def __init__(self, base=32, emb_dim=128):
        super().__init__()
        e = emb_dim
        self.temb = TimeEmbed(e)
        self.tmlp = nn.Sequential(nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e))
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.d1a = ResBlock(base, base, e)
        self.d1b = ResBlock(base, base, e)
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1)      # 28 -> 14
        self.d2a = ResBlock(base, base * 2, e)
        self.d2b = ResBlock(base * 2, base * 2, e)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)  # 14 -> 7
        self.mid_a = ResBlock(base * 2, base * 2, e)
        self.mid_b = ResBlock(base * 2, base * 2, e)
        self.u2a = ResBlock(base * 4, base * 2, e)
        self.u2b = ResBlock(base * 2, base * 2, e)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")          # 7 -> 14
        self.u1a = ResBlock(base * 3, base, e)
        self.u1b = ResBlock(base, base, e)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")          # 14 -> 28
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)
    def forward(self, x, t):
        emb = self.tmlp(self.temb(t))
        h0 = self.in_conv(x)
        h1 = self.d1b(self.d1a(h0, emb), emb)          # 28, base
        h2 = self.d2b(self.d2a(self.down1(h1), emb), emb)  # 14, 2base
        m = self.mid_b(self.mid_a(self.down2(h2), emb), emb)  # 7, 2base
        u2 = self.u2b(self.u2a(torch.cat([self.up2(m), h2], 1), emb), emb)
        u1 = self.u1b(self.u1a(torch.cat([self.up1(u2), h1], 1), emb), emb)
        return self.out_conv(F.silu(self.out_norm(u1)))

# ---------------- Training ----------------
def train(max_steps=3000, time_budget_min=45, batch=64, lr=2e-4, T=500):
    imgs, labels = load_mnist()
    sched = cosine_alphas(T)
    model = UNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = imgs.shape[0]
    t0 = time.time()
    step = 0
    log_path = "train_log.txt"
    while step < max_steps and (time.time() - t0) < time_budget_min * 60:
        idx = torch.randint(0, n, (batch,))
        x0 = imgs[idx]
        t = torch.randint(0, T, (batch,))
        noise = torch.randn_like(x0)
        ab = sched["alpha_bar"][t]
        x_t = (ab.sqrt()) * x0 + (1 - ab).sqrt() * noise   # forward noising
        pred = model(x_t, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 50 == 0:
            elapsed = time.time() - t0
            msg = f"step {step}/{max_steps} | loss {loss.item():.4f} | {elapsed/60:.1f} min elapsed"
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
        if step % 500 == 0:
            torch.save(model.state_dict(), f"checkpoint_{step}.pt")
            print(f"saved checkpoint_{step}.pt", flush=True)
    torch.save(model.state_dict(), "visionai_weights.pt")
    print("TRAINING COMPLETE — final weights: visionai_weights.pt", flush=True)
    return model, sched

# ---------------- Sampling ----------------
@torch.no_grad()
def sample(model, sched, T=500, n=16, eta=0.0):
    model.eval()
    x = torch.randn(n, 1, 28, 28)
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, dtype=torch.long)
        ab, a, b = sched["alpha_bar"][t], sched["alpha"][t], sched["beta"][t]
        pred_noise = model(x, t_batch)
        x0_pred = (x - (1 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        x = (ab.sqrt() * x0_pred + (1 - ab).sqrt() * pred_noise) * a.sqrt() + (1 - a).sqrt() * torch.randn_like(x)
        if t % 100 == 0:
            print(f"denoising step {t}", flush=True)
    return x

def save_grid(x, path="samples.png"):
    from PIL import Image
    n = x.shape[0]; side = int(math.isqrt(n))
    canvas = Image.new("L", (28 * side, 28 * side))
    for i in range(side * side):
        img = ((x[i, 0].clamp(-1, 1) + 1) * 127.5).byte().numpy()
        canvas.paste(Image.fromarray(img), ((i % side) * 28, (i // side) * 28))
    canvas.save(path)
    print(f"saved {path}", flush=True)

if __name__ == "__main__":
    model, sched = train()
    x = sample(model, sched)
    save_grid(x)
    print("DONE — VISIONAI model generated its first images from pure noise", flush=True)
