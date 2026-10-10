"""VISIONAI on-demand generator: sample any class, upscale to big PNG.
Usage: python3 make_image.py <class> [count] [size]   e.g. python3 make_image.py cat 2 256"""
import sys, time
import torch
from PIL import Image, ImageFilter, ImageEnhance
from cifar_ddpm import CondUNet, CLASSES, T
from ddpm import cosine_alphas

def main():
    cls_name = sys.argv[1].lower() if len(sys.argv) > 1 else "cat"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    if cls_name not in CLASSES:
        print(f"Unknown class '{cls_name}'. Choose from: {', '.join(CLASSES)}"); return
    ci = CLASSES.index(cls_name)

    sched = cosine_alphas(T)
    model = CondUNet()
    ck = torch.load("cifar_ckpt.pt", weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"model loaded (step {ck.get('step')}), generating {count} x {cls_name}...", flush=True)

    with torch.no_grad():
        img = torch.randn(count, 3, 32, 32)
        cls = torch.full((count,), ci, dtype=torch.long)
        for t in reversed(range(T)):
            tb = torch.full((count,), t, dtype=torch.long)
            ab, a = sched["alpha_bar"][t], sched["alpha"][t]
            pn = model(img, tb, cls)
            x0 = ((img - (1-ab).sqrt()*pn) / ab.sqrt()).clamp(-1, 1)
            img = (ab.sqrt()*x0 + (1-ab).sqrt()*pn) * a.sqrt() + (1-a).sqrt() * torch.randn_like(img)

    outs = []
    for i in range(count):
        arr = ((img[i].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
        big = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
        big = big.filter(ImageFilter.SMOOTH).filter(ImageFilter.UnsharpMask(radius=4, percent=60, threshold=3))
        big = ImageEnhance.Contrast(big).enhance(1.05)
        path = f"gen_{cls_name}_{i}.png"
        big.save(path)
        outs.append(path)
        print(f"saved {path} ({size}x{size})", flush=True)

if __name__ == "__main__":
    main()
