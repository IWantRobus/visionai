"""Chunked, resumable trainer — survives sandbox process kills."""
import sys, time, torch
import torch.nn.functional as F
from ddpm import load_mnist, cosine_alphas, UNet

MAX_STEPS = 2500
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
T = 500

imgs, labels = load_mnist()
sched = cosine_alphas(T)
model = UNet()
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
step = 0
try:
    ck = torch.load("checkpoint_last.pt", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    print(f"resumed from step {step}", flush=True)
except FileNotFoundError:
    print("starting fresh", flush=True)

n = imgs.shape[0]; t0 = time.time()
while step < MAX_STEPS and (time.time() - t0) < MINUTES * 60:
    idx = torch.randint(0, n, (64,))
    x0 = imgs[idx]; t = torch.randint(0, T, (64,))
    noise = torch.randn_like(x0)
    ab = sched["alpha_bar"][t]
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    loss = F.mse_loss(model(x_t, t), noise)
    opt.zero_grad(); loss.backward(); opt.step()
    step += 1
    if step % 50 == 0:
        print(f"step {step}/{MAX_STEPS} | loss {loss.item():.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
    if step % 100 == 0:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, "checkpoint_last.pt")

torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, "checkpoint_last.pt")
print(f"CHUNK COMPLETE at step {step}", flush=True)
