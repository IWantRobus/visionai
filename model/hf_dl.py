"""Verified download + conversion of CIFAR-10 train split from Hugging Face."""
import io, os
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="uoft-cs/cifar10", repo_type="dataset",
                       filename="plain_text/train-00000-of-00001.parquet")
print("downloaded:", path, flush=True)

import pyarrow.parquet as pq
from PIL import Image
import torch

table = pq.read_table(path)
imgs = table.column("img").to_pylist()
labels = table.column("label").to_pylist()
print(f"{len(imgs)} rows", flush=True)

x = torch.empty(len(imgs), 3, 32, 32, dtype=torch.uint8)
for i, rec in enumerate(imgs):
    img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
    x[i] = torch.frombuffer(img.tobytes(), dtype=torch.uint8).view(3, 32, 32)
    if i % 10000 == 0:
        print(f"converted {i}", flush=True)
torch.save({"x": x, "y": torch.tensor(labels)}, "cifar_data.pt")
print("SAVED cifar_data.pt", flush=True)
open("DATA_DONE", "w").write("ok")
