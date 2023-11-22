# %%

# %%
import torch
import torchio as tio

torch.set_grad_enabled(False);

# %%
device = torch.device('cuda')

# %%
transforms = [
    tio.ToCanonical(),
    tio.Resample(),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
]
transform = tio.Compose(transforms)

# %%
dataset = tio.datasets.IXI(
    '/tmp/data_jvol',
    transform=transform,
    modalities=['T1'],
    download=True,
)

# %%
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=6,
    pin_memory=True,
)

# %%
repo = 'fepegar/highresnet'
model_name = 'highres3dnet'
model = torch.hub.load(repo, model_name, pretrained=True, trust_repo=True)
model.to(device).eval();

# %%
import torch.nn as nn
model.block[-1] = nn.Identity()

# %%
from tqdm.auto import tqdm
all_features = []
for subject in tqdm(loader):
    image = subject['T1'][tio.DATA].to(device)
    features = model(image).mean(dim=(-3, -2, -1)).cpu()
    all_features.append(features)
all_features = torch.cat(all_features)

torch.save(all_features, '/home/fperezgarcia/git/jvol/features_lossless.pth')
