"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import os
import pickle
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from aurora import AuroraPretrained, Batch, Metadata

# --- 1. Define the Dataset ---
class MultivarDataset(Dataset):
    def __init__(self, data_path, num_timesteps=124100):
        # 124,100 = (2025 - 1940) * 1460 
        self.data = np.memmap(
            data_path, 
            dtype=np.float32, 
            mode='r', 
            shape=(num_timesteps, 5, 721, 1440)
        )
        
    def __len__(self):
        # Subtract 1 because we need a target for t+1
        return len(self.data) - 1
        
    def __getitem__(self, idx):
        # .copy() forces the slice into RAM, preventing memory leaks across dataloader workers
        input_data = torch.from_numpy(self.data[idx].copy())
        target_data = torch.from_numpy(self.data[idx + 1].copy())
        return input_data, target_data

# --- 2. Helper to Build Aurora Batches ---
def create_aurora_batch(tensor_data, static_dict, device="cuda"):
    """
    Maps the (B, 5, 721, 1440) tensor back to Aurora's dict structure.
    Channel map from make_trainset.py: 
      0: slp (msl)
      1: u (500)
      2: v (500)
      3: t (925)
      4: q (500)
    """
    tensor_data = tensor_data.to(device)
    B = tensor_data.shape[0]
    
    # Surface variables: (B, Time, Lat, Lon)
    surf_vars = {
        "msl": tensor_data[:, 0:1, :, :] # Keep dimension 1 as Time (size 1)
    }
    
    # Atmospheric variables: (B, Time, Levels, Lat, Lon)
    # Level 0 = 500 hPa, Level 1 = 925 hPa
    u = torch.zeros((B, 1, 2, 721, 1440), device=device)
    v = torch.zeros((B, 1, 2, 721, 1440), device=device)
    t = torch.zeros((B, 1, 2, 721, 1440), device=device)
    q = torch.zeros((B, 1, 2, 721, 1440), device=device)
    
    # Fill 500 hPa variables
    u[:, 0, 0, :, :] = tensor_data[:, 1, :, :]
    v[:, 0, 0, :, :] = tensor_data[:, 2, :, :]
    q[:, 0, 0, :, :] = tensor_data[:, 4, :, :]
    
    # Fill 925 hPa variables
    t[:, 0, 1, :, :] = tensor_data[:, 3, :, :]
    
    atmos_vars = {"u": u, "v": v, "t": t, "q": q}
    
    # Static variables: (Lat, Lon)
    static_vars = {
        "lsm": torch.from_numpy(static_dict["lsm"]).to(device),
        "z": torch.from_numpy(static_dict["z"]).to(device),
        "slt": torch.from_numpy(static_dict["slt"]).to(device),
    }
    
    metadata = Metadata(
        lat=torch.linspace(90, -90, 721),
        lon=torch.linspace(0, 360, 1440 + 1)[:-1],
        time=[datetime(2020, 6, 1, 12, 0)] * B, # Dummy time for inference
        atmos_levels=(500, 925),
    )
    
    return Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)

# --- 3. Custom Loss Function ---
def calc_loss(pred: Batch, target: Batch) -> torch.Tensor:
    """Calculates MSE ONLY on the valid pressure levels."""
    loss = F.mse_loss(pred.surf_vars["msl"], target.surf_vars["msl"])
    
    # 500 hPa variables
    loss += F.mse_loss(pred.atmos_vars["u"][:, :, 0], target.atmos_vars["u"][:, :, 0])
    loss += F.mse_loss(pred.atmos_vars["v"][:, :, 0], target.atmos_vars["v"][:, :, 0])
    loss += F.mse_loss(pred.atmos_vars["q"][:, :, 0], target.atmos_vars["q"][:, :, 0])
    
    # 925 hPa variables
    loss += F.mse_loss(pred.atmos_vars["t"][:, :, 1], target.atmos_vars["t"][:, :, 1])
    
    return loss

# --- 4. Main Training Loop ---
if __name__ == "__main__":
    out_dir = '/mnt/data/sonia/aurora-data/train'
    data_path = os.path.join(out_dir, 'train_data.dat')
    static_path = os.path.join(out_dir, 'aurora-0.25-static.pickle')

    print("Loading static variables...")
    with open(static_path, "rb") as f:
        static_dict = pickle.load(f)

    # Note: Fixed the surf_vars tuple syntax (added a comma)
    print("Initializing Aurora...")
    model = AuroraPretrained(
        autocast=True,
        surf_vars=("msl",), 
        static_vars=("lsm", "z", "slt"),
        atmos_vars=("u", "v", "t", "q")
    )
    model.load_checkpoint(strict=False)
    model.configure_activation_checkpointing()
    model.train()
    model = model.to("cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    dataset = MultivarDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    print("Starting training loop...")
    pbar = tqdm(dataloader, desc="Training")
    for step, (inputs, targets) in enumerate(pbar):
        print(f"Step {step}")
        
        # 1. Format inputs and targets into Batch objects
        input_batch = create_aurora_batch(inputs, static_dict, device="cuda")
        target_batch = create_aurora_batch(targets, static_dict, device="cuda")
        
        # 2. Normalize both using Aurora's internal statistics
        input_batch = input_batch.normalise(model.surf_stats)
        target_batch = target_batch.normalise(model.surf_stats)

        opt.zero_grad()
        
        # 3. Forward pass (predicts t+1)
        prediction = model(input_batch)
        
        # 4. Calculate loss on normalized tensors
        loss_value = calc_loss(prediction, target_batch)
        
        loss_value.backward()
        opt.step()
        
        pbar.set_postfix(loss=f"{loss_value.item():.4f}")
            
            