import torch
import torch.nn.functional as F
from tqdm import tqdm

def get_softmax_out(model, loader, device, is_dac=False):
    softmax_out = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            if is_dac:
                softmax_out.append(F.softmax(model(data)[:,:-1], dim=1))
            else:
                softmax_out.append(torch.exp(model(data)))
    return torch.cat(softmax_out).cpu().numpy()

def get_pseudo_labels(loader):
    total_out = []
    loop = tqdm(loader, leave=True)
    with torch.no_grad():
        for _, y in loop:
            total_out.append(y)
    return torch.cat(total_out).cpu().numpy()