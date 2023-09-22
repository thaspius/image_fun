import torch

from PIL import Image
from lavis.models import load_model_and_preprocess

print("Is cuda available: " + torch.cuda.is_available())


model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device="cuda", is_eval=True)

