"""VISIONAI CORE v2 — class-conditional DDPM on CIFAR-10 (60,000 real internet-collected photos).
Conditioning mechanism: the same principle text-to-image is built on, driven by class embeddings.
Self-resuming trainer: survives process kills via checkpoints."""
import math, os, pickle, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm import cosine_alphas, TimeEmbed, ResBlock

torch.manual_seed(42)
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def load_cifar():
    if os.path.exists("cifar_data.pt"):
        d = torch.load("cifar_data.pt", weights_only=False)
        x = d["x"].float() / 127.5 - 1.0
        y = d["y"]
        print(f"Loaded {x.shape[0]} real internet-collected photos", flush=True)
        return x, y
    imgs, labels = [], []
    for i in range(1, 6):
        with open(f"cifar-10-batches-py/data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs.append(torch.tensor(d[b"data"]).view(-1, 3, 32, 32).float())
        labels.append(torch.tensor(d[b"labels"]))
    x = torch.cat(imgs) / 127.5 - 1.0
    y = torch.cat(labels)
    print(f"Loaded {x.shape[0]} real internet-collected photos", flush=True)
    return x, y

class CondUNet(nn.Module):
    """3-channel, class-conditioned U-Net: 32 -> 16 -> 8."""
    def __init__(self, base=48, emb_dim=128, n_cls=10):
        super().__init__()
        e = emb_dim
        self.temb = TimeEmbed(e)
        self.tmlp = nn.Sequential(nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e))
        self.cls = nn.Embedding(n_cls, e)
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        self.d1a = ResBlock(base, base, e); self.d1b = ResBlock(base, base, e)
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.d2a = ResBlock(base, base*2, e); self.d2b = ResBlock(base*2, base*2, e)
        self.down2 = nn.Conv2d(base*2, base*2, 3, stride=2, padding=1)
        self.m1 = ResBlock(base*2, base*2, e); self.m2 = ResBlock(base*2, base*2, e)
        self.u2a = ResBlock(base*4, base*2, e); self.u2b = ResBlock(base*2, base*2, e)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.u1a = ResBlock(base*3, base, e); self.u1b = ResBlock(base, base, e)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)
    def forward(self, x, t, cls):
        emb = self.tmlp(self.temb(t)) + self.cls(cls)
        h1 = self.d1b(self.d1a(self.in_conv(x), emb), emb)
        h2 = self.d2b(self.d2a(self.down1(h1), emb), emb)
        m = self.m2(self.m1(self.down2(h2), emb), emb)
        u2 = self.u2b(self.u2a(torch.cat([self.up2(m), h2], 1), emb), emb)
        u1 = self.u1b(self.u1a(torch.cat([self.up1(u2), h1], 1), emb), emb)
        return self.out_conv(F.silu(self.out_norm(u1)))

T = 500
MAX_STEPS = int(os.environ.get("MAX_STEPS", "4000"))

def train(minutes):
    x, y = load_cifar()
    sched = cosine_alphas(T)
    model = CondUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    step = 0
    try:
        ck = torch.load("cifar_ckpt.pt", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    except FileNotFoundError:
        print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    n = x.shape[0]; t0 = time.time()
    while step < MAX_STEPS and (time.time() - t0) < minutes * 60:
        idx = torch.randint(0, n, (32,))
        x0, cls = x[idx], y[idx]
        t = torch.randint(0, T, (32,))
        noise = torch.randn_like(x0)
        ab = sched["alpha_bar"][t]
        loss = F.mse_loss(model(ab.sqrt()*x0 + (1-ab).sqrt()*noise, t, cls), noise)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 25 == 0:
            msg = f"step {step}/{MAX_STEPS} | loss {loss.item():.4f} | {(time.time()-t0)/60:.1f} min"
            print(msg, flush=True)
        if step % 100 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, "cifar_ckpt.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, "cifar_ckpt.pt")
    print(f"CHUNK DONE step {step}", flush=True)
    return step >= MAX_STEPS

@torch.no_grad()
def sample(n_per_class=2, path="cifar_samples.png"):
    # no dataset load needed for sampling
    sched = cosine_alphas(T)
    model = CondUNet()
    ck = torch.load("cifar_ckpt.pt", weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    n = n_per_class * 10
    cls = torch.arange(10).repeat_interleave(n_per_class)
    img = torch.randn(n, 3, 32, 32)
    for t in reversed(range(T)):
        tb = torch.full((n,), t, dtype=torch.long)
        ab, a = sched["alpha_bar"][t], sched["alpha"][t]
        pn = model(img, tb, cls)
        x0 = ((img - (1-ab).sqrt()*pn) / ab.sqrt()).clamp(-1, 1)
        img = (ab.sqrt()*x0 + (1-ab).sqrt()*pn) * a.sqrt() + (1-a).sqrt() * torch.randn_like(img)
        if t % 250 == 0: print(f"denoise {t}", flush=True)
    from PIL import Image
    rows, cols = 10, n_per_class
    canvas = Image.new("RGB", (cols*32, rows*32), (10,10,20))
    for i in range(n):
        arr = ((img[i].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
        r, c = i // n_per_class, i % n_per_class
        canvas.paste(Image.fromarray(arr), (c*32, r*32))
    canvas = canvas.resize((cols*32*4, rows*32*4), Image.NEAREST)
    canvas.save(path)
    print(f"saved {path}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        sample()
    else:
        done = train(float(sys.argv[1]) if len(sys.argv) > 1 else 8)
        if done:
            open("TRAIN_DONE", "w").write("ok")
