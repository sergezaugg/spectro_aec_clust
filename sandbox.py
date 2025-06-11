





import torch
from utils import AutoencoderExtract
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




pa = "D:/xc_real_projects/trained_models/20250610_002647_encoder_script_gen_B0L.pth"


model_enc = torch.jit.load(pa)
model_enc = model_enc.to(device)
_ = model_enc.eval()



