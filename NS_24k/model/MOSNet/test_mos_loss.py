import numpy as np
import torch
import soundfile as sf
from model.MOSNet.mosnet import MOSNet
import torch.nn.functional as F
device = "cuda"

weights_path = "mosnet16_torch.pt"

# Initialize model and load weights:
mos_model = MOSNet(device=device)
mos_model.load_state_dict(torch.load(weights_path))
mos_model.to(device)
mos_model.eval()

# Load speech audio sample:
speech_path = "sample.wav"
y, _ = sf.read(speech_path)

with torch.no_grad():
    y_in = torch.from_numpy(y).unsqueeze(0).to(device)
    test_x_in = torch.ones_like(y_in) * 0.5
    test_x_in = test_x_in.to(device)
    # mos_average, mos_per_frame = mos_model(y_in)

    # Compute L1 loss on feature maps:
    ftrs_x = mos_model.getFtrMaps(test_x_in)
    ftrs_ref = mos_model.getFtrMaps(y_in)
loss = 0
for i in range(len(ftrs_x)):
    loss += F.l1_loss(ftrs_x[i], ftrs_ref[i]).mean()
loss /= len(ftrs_x)

print(loss)

