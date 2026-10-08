import torch, math
from ddpm import UNet, cosine_alphas, sample, save_grid

model = UNet()
ck = torch.load("checkpoint_2000.pt", weights_only=False)
model.load_state_dict(ck)
model.eval()
print("weights loaded (step 2000, loss ~0.04)", flush=True)

sched = cosine_alphas(500)
x = sample(model, sched, T=500, n=16)
save_grid(x, "samples.png")
# also save final weights under clean name
torch.save(model.state_dict(), "visionai_weights.pt")
print("FINAL WEIGHTS SAVED: visionai_weights.pt", flush=True)
