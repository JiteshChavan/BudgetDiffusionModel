import torch


import re
from pathlib import Path
import torch
from PIL import Image

from micro_diffusion.models.model import create_latent_diffusion

batch_size = 10
captions = ['cinnamon bun', 'violet evergarden flower', 'aurora borealis', 'lotus in water color style'] * 4

# VAE compression factor 8 -> pixel space res = 512
model = create_latent_diffusion(latent_res=64, in_channels=4, pos_interp_scale=2.0).to('cuda')
path_to_ckpt = "REPLACE_WITH_LOCAL_PATH_TO_CKPT"
model.dit.load_state_dict(torch.load(path_to_ckpt, map_location='cuda'))
model.eval()

batches = len(captions) // batch_size

out_dir = "renders"

global_idx = 0
with torch.inference_mode():
    for render_batch in range(batches):
        prompt_batch = captions[render_batch * batch_size : (render_batch + 1) * batch_size]

        images, frames = model.generate(
            prompt=prompt_batch,
            num_inference_steps=30,
            guidance_scale=5.0,
            seed=1997 + render_batch,
            return_frames=True,
            save_every=1,
        )

        save_renders(out_dir, prompt_batch, images, frames, global_start_idx=global_idx)
        global_idx += len(prompt_batch)


def get_unique_folder_name(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "default"

def tensor_to_image(x_chw: torch.Tensor, path: Path):
    # x_chw: [3,H,W] in [0,1]
    x = (x_chw.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(x).save(path)

def save_renders(
    out_root: str,
    prompt_batch: list[str],
    images: torch.Tensor,          # (B, C, H, W)
    frames: list[torch.Tensor],    # list of (B, C, H, W)
    global_start_idx: int = 0
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    B = len(prompt_batch)
    images = images.detach().cpu()

    for b in range(B):
        global_i = global_start_idx + b
        folder = out_root / f"{global_i:04d}_{get_unique_folder_name(prompt_batch[b])}"
        folder.mkdir(parents=True, exist_ok=True)

        # save frames
        for t, fr in enumerate(frames):
            fr_b = fr[b]  # [3,H,W]
            tensor_to_image(fr_b, folder / f"frame_{t:03d}.png")

        # save final
        tensor_to_image(images[b], folder / "final.png")