import torch

data = torch.load("speakers/neil_gaiman.pt", map_location="cpu")
print(data)