
import torch
import os

path_trained_models = "D:/xc_real_projects/trained_models"

model_enc = torch.load( os.path.join(path_trained_models, '20250606_162046_encoder_model_gen_B21.pth'), weights_only = False)
model_dec = torch.load( os.path.join(path_trained_models, '20250606_162046_decoder_model_gen_B21.pth'), weights_only = False)




model_enc_scripted = torch.jit.script(model_enc) # Export to TorchScript
model_enc_scripted.save(os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pt'))


model_dec_scripted = torch.jit.script(model_dec) # Export to TorchScript
model_dec_scripted.save(os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pt'))





