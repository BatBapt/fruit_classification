import torch


BASE_PATH = "D:/Programmation/IA/datas/fruit_classif/Fruit_dataset"

TORCH_MODEL_PATH = "D:/models/torch/hub"
torch.hub.set_dir(TORCH_MODEL_PATH) # Uncomment this line to set the directory to download pytorch models

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')