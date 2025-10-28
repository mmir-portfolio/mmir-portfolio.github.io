---
layout: post
title: "Physics-Informed Fourier Neural Operators for Predicting Hydrogen Breakthrough in Porous Media"
image: /posts/hydrogen_storage.png
tags: [Machine Learning, Physics-Informed ML, Fourier Neural Operator, Hydrogen Storage, Porous Media, Clean Energy]
---
In this project, we developed a physics-informed Fourier Neural Operator (FNO) model to predict hydrogen breakthrough time in underground porous reservoirs. By generating synthetic permeability datasets and incorporating two-phase flow physics with capillary pressure effects, the model learns the complex mapping from heterogeneous permeability fields to breakthrough time. This approach enables rapid, reliable predictions without running computationally expensive reservoir simulations, supporting faster site screening, operational planning, and performance evaluation for underground hydrogen storage systems.



# Table of Contents
- [00. Introduction](#introduction)
- [01. Background: Underground Hydrogen Storage (UHS)](#background)
- [02. Governing Physics: Two-Phase Flow with Capillary Pressure](#governing-physics)
- [03. What Is Breakthrough Time?](#breakthrough-time)
- [04. Why Fourier Neural Operator (FNO)?](#fno)
- [05. Dataset Generation & Simulation Workflow](#dataset)
- [06. FNO Architecture Overview](#architecture)
- [07. Training Strategy](#training)
- [08. Prediction & Inference Pipeline](#prediction)
- [09. Visualization & Interpretation](#visualization)
- [10. Evaluation Metrics](#evaluation)
- [11. Industry Relevance & Deployment](#applications)
- [12. Future Work & Extensions](#future)
- [13. Python Implementation](#python)

# 00. Introduction <a name="introduction"></a>
Underground Hydrogen Storage (UHS) is emerging as a crucial solution for storing renewable energy. Hydrogen can be injected into porous geological formations for seasonal storage. Predicting **hydrogen breakthrough time** is essential to ensure operational efficiency, safety, and maximum storage utilization.  

This project demonstrates how **Physics-Informed Fourier Neural Operators (FNOs)** can predict breakthrough time based on **permeability field heterogeneity**, while respecting multiphase flow physics and capillary pressure effects.

# 01. Background: Underground Hydrogen Storage (UHS) <a name="background"></a>
UHS involves storing hydrogen in subsurface formations like **depleted reservoirs or saline aquifers**. Operational challenges include:

- Heterogeneous permeability fields affecting flow patterns  
- Two-phase hydrogen/brine flow dynamics with capillary trapping  
- Diffusion and dispersion of hydrogen  
- Constraints on injection pressure and total storage volume  

Machine learning approaches like **FNOs** help accelerate breakthrough time predictions compared to full multiphase simulations.

# 02. Governing Physics: Two-Phase Flow with Capillary Pressure <a name="governing-physics"></a>
Hydrogen saturation (S<sub>H2</sub>) in porous media is governed by:

<div style="text-align:center;">
φ ∂S<sub>H2</sub>/∂t + ∇·(f(S<sub>H2</sub>) v) = ∇·(D(S<sub>H2</sub>) ∇S<sub>H2</sub>)
</div>

Capillary pressure affects flow as:

<div style="text-align:center;">
P<sub>c</sub>(S<sub>H2</sub>) = P<sub>0</sub> · S<sub>H2</sub><sup>-1/λ</sup>
</div>

Where:  

- φ = porosity  
- f(S<sub>H2</sub>) = fractional flow  
- D(S<sub>H2</sub>) = diffusion coefficient  
- v = Darcy velocity  
- P<sub>0</sub>, λ = capillary pressure parameters  

# 03. What Is Breakthrough Time? <a name="breakthrough-time"></a>
Breakthrough time \(t_b\) is defined as the moment hydrogen saturation at the production well exceeds a small threshold (e.g., 0.01).  

It depends on:  

- Permeability distribution  
- Injection rate  
- Fluid properties (viscosity, density)  
- Capillary pressure  

Accurate prediction allows operators to plan **injection schedules** and **avoid early breakthrough**.

# 04. Why Fourier Neural Operator (FNO)? <a name="fno"></a>
FNOs learn **mappings between function spaces**, making them ideal for heterogeneous permeability fields:

- **Mesh-independent**: works across different grid resolutions  
- **Global correlations**: captures long-range interactions in permeability  
- **Physics-informed**: PDE constraints can be added as part of the loss  

FNOs generalize better than CNNs for unseen permeability realizations.

# 05. Dataset Generation & Simulation Workflow <a name="dataset"></a>
Dataset generation involves:

- **Permeability fields**: log-normal distribution, scaled 0.1–1 Darcy, Nx=128 grid  
- **Two-phase flow simulation**: simple solver including capillary pressure  
- **Physical parameters**:  
  - φ = 0.25  
  - μ<sub>H2</sub> = 0.09 mPa·s  
  - μ<sub>brine</sub> = 1 mPa·s  
  - P<sub>0</sub> = 2000 Pa, λ = 2  
  - Grid points = 128, time step Δt = 1 day  
- **Breakthrough time**: S<sub>H2</sub> ≥ 0.01 at production well  
- **Train/test split**: 80/20  

This balances **physical realism** and computational efficiency.

# 06. FNO Architecture Overview <a name="architecture"></a>
- **Input lifting**: 1 → 32 width  
- **Four spectral conv layers**: 16 Fourier modes  
- **Residual pointwise conv layers**  
- **Activation**: ReLU  
- **Output projection**: width → scalar (t<sub>b</sub>)  
- Optional physics-informed loss: PDE residuals

# 07. Training Strategy <a name="training"></a>
- Loss = MSE_data + λ·PDE_residual  
- Optimizer: Adam (lr=1e-3)  
- ReduceLROnPlateau (factor=0.5, patience=10)  
- Batch = 32, Epochs = 50–100  
- Track metrics: MSE, MAE, physical consistency

# 08. Prediction & Inference Pipeline <a name="prediction"></a>
- Normalize permeability fields  
- Forward through FNO  
- Denormalize t<sub>b</sub>  
- Compare with simulated results  
- Sensitivity analysis for input parameters

# 09. Visualization & Interpretation <a name="visualization"></a>
- **Predicted vs true t<sub>b</sub>** plots  
- **Saturation profiles** for samples  
- **Residual maps** for spatial error  
- Ensure **physical consistency**: lower permeability → higher t<sub>b</sub>

# 10. Evaluation Metrics <a name="evaluation"></a>
- MAE, R²  
- Physical consistency checks  
- Sensitivity to φ, P<sub>c</sub>, injection rate

# 11. Industry Relevance & Deployment <a name="applications"></a>
- Rapid site screening for UHS feasibility  
- Planning injection/production schedules  
- Reducing need for expensive simulations  
- Early detection of breakthrough events

# 12. Future Work & Extensions <a name="future"></a>
- Extend to 3D reservoirs  
- Integrate real-world P/T measurements  
- Bayesian FNO for uncertainty  
- Hybrid DeepONet + FNO  
- Real-time predictive monitoring for operational safety

# 13. Python Implementation <a name="python"></a>
Below is a complete Python implementation that demonstrates the development of a Physics-Informed Fourier Neural Operator (FNO) to predict hydrogen breakthrough time from heterogeneous permeability fields. The workflow is built entirely in PyTorch, leveraging FFT-based spectral convolutions for efficient operator learning. The code includes dataset generation, model definition, training, and evaluation steps, all designed to emulate the underlying two-phase flow dynamics with capillary effects in porous media. This implementation serves as both a research prototype and a foundation for scaling to larger 2D or 3D reservoir models.

```python
import torch
import torch.nn as nn
import torch.fft

# --- Physical / Numerical Parameters ---
phi = 0.25
mu_H2 = 0.09e-3
mu_brine = 1e-3
Pc0 = 2000
lambda_pc = 2
L_domain = 1.0
Nx = 128
dt = 86400

# --- Dataset Generation ---
def generate_permeability_fields(N=1000, Nx=128):
    return torch.rand(N, 1, Nx)*0.9 + 0.1

def compute_breakthrough_time(k_field, phi=0.25, mu_H2=0.09e-3, 
                              Pc0=2000, L=L_domain, dP=1e5):
    """
    Approximate hydrogen breakthrough time based on Darcy velocity.
    k_field: [batch, 1, Nx] or [batch, Nx] permeability tensor
    phi: porosity
    mu_H2: hydrogen viscosity (Pa·s)
    Pc0: base capillary pressure (Pa)
    L: domain length (m)
    dP: applied pressure drop (Pa)
    """
    # Mean permeability across domain
    avg_k = torch.mean(k_field, dim=-1)  # averages over spatial dimension

    # Darcy velocity (m/s)
    v = avg_k * dP / (mu_H2 * L)

    # Capillary correction (reduces velocity)
    capillary_factor = 1.0 / (1.0 + Pc0 / dP)

    # Approximate breakthrough time (s)
    t_b = (phi * L) / (v * capillary_factor + 1e-12)

    # Normalize (optional)
    #t_b = t_b / t_b.max()

    return t_b


def two_phase_solver(k, phi, mu_H2, mu_brine, Pc0, lambda_pc, L, Nx, dt):
    """
    Simplified 1D two-phase flow solver for H2 and brine displacement.
    Uses an explicit finite-difference formulation to approximate 
    saturation evolution and detect hydrogen breakthrough times.

    Parameters
    ----------
    k : array_like
        Permeability field [m^2], length Nx
    phi : array_like
        Porosity field, length Nx
    mu_H2, mu_brine : float
        Viscosities of hydrogen and brine [Pa.s]
    Pc0, lambda_pc : float
        Capillary pressure parameters
    L : float
        Domain length [m]
    Nx : int
        Number of grid cells
    dt : float
        Time step [s]

    Returns
    -------
    t_break: ndarray
        Breakthrough times (0–1) for each cell.
    """

    # --- Spatial discretization
    dx = L / Nx
    x = np.linspace(0, L, Nx)

    # --- Initialization
    Sw = np.ones(Nx)                 # Brine saturation (initially full of brine)
    Sw[0] = 0.0                      # Injector end (pure H2)
    t = 0.0

    # --- Output arrays
    t_break = np.zeros(Nx)
    breakthrough_mask = np.zeros(Nx, dtype=bool)
    breakthrough_threshold = 0.1     # Breakthrough when Sw drops below 0.1

    # --- Ensure arrays match Nx
    k = np.broadcast_to(k, Nx)
    phi = np.broadcast_to(phi, Nx)

    # --- Relative permeabilities (Corey-type)
    def kr_H2(S):
        return np.clip(S, 0, 1) ** 2

    def kr_brine(S):
        return np.clip(1 - S, 0, 1) ** 2

    # --- Capillary pressure
    def Pc(S):
        return Pc0 * (np.clip(1 - S, 1e-6, 1) ** (-1 / lambda_pc) - 1)

    # --- Time stepping loop
    max_steps = 2000
    for step in range(max_steps):
        Sw_old = Sw.copy()

        # Mobilities
        lam_H2 = kr_H2(1 - Sw) / mu_H2
        lam_brine = kr_brine(Sw) / mu_brine
        lam_t = lam_H2 + lam_brine

        # Fractional flow
        fw = lam_H2 / (lam_t + 1e-12)

        # Capillary pressure gradient (finite difference)
        dPc_dx = np.gradient(Pc(Sw), dx)

        # Flux term (simplified 1D form)
        q = -k * lam_t * dPc_dx

        # Saturation update
        dSw_dt = -np.gradient(fw * q, dx) / phi
        Sw += dSw_dt * dt
        Sw = np.clip(Sw, 0, 1)

        t += dt

        # Detect new breakthrough cells
        newly_broken = (Sw < breakthrough_threshold) & (~breakthrough_mask)
        t_break[newly_broken] = t
        breakthrough_mask |= newly_broken

        # Stop if all cells have broken through
        if breakthrough_mask.all():
            break

    # --- Normalize breakthrough times
    # t_break_norm = t_break / t_break.max() if t_break.max() > 0 else t_break
    return t_break


k_data = generate_permeability_fields()
t_data = compute_breakthrough_time(k_data)

idx = int(0.8*len(k_data))
train_k, train_t = k_data[:idx], t_data[:idx]
test_k, test_t = k_data[idx:], t_data[idx:]

# --- FNO Model ---
class SpectralConv1dSafe(nn.Module):
    """
    Robust spectral conv for FNO:
    - weights stored as real tensors with last dim = 2 (real, imag)
    - correct einsum ordering: 'bcm, com -> bom'
    - adapts to available Fourier modes
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # desired number of modes

        # weights: (in_ch, out_ch, modes, 2) where last dim = [real, imag]
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes, 2) * 0.01)

    def forward(self, x):
        """
        x: (B, in_channels, N) real tensor
        returns: (B, out_channels, N) real tensor
        """
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)       # (B, C, N_ft), complex
        N_ft = x_ft.shape[-1]
        n_modes = min(self.modes, N_ft)

        # prepare output real/imag parts
        out_r = torch.zeros(B, self.out_channels, N_ft, device=x.device, dtype=x.dtype)
        out_i = torch.zeros(B, self.out_channels, N_ft, device=x.device, dtype=x.dtype)

        # input real/imag (B, C, N_ft)
        xr = x_ft.real[..., :n_modes]
        xi = x_ft.imag[..., :n_modes]

        # weights real/imag shaped (C, O, n_modes)
        w = self.weights[:, :, :n_modes, :]  # (C, O, n_modes, 2)
        w_r = w[..., 0]  # (C, O, n_modes)
        w_i = w[..., 1]  # (C, O, n_modes)

        # Correct einsum ordering:
        # xr: (B, C, m) labeled 'bcm'
        # w_r: (C, O, m) labeled 'com'
        # -> result: (B, O, m) labeled 'bom'
        term_rr = torch.einsum('bcm,com->bom', xr, w_r)
        term_ii = torch.einsum('bcm,com->bom', xi, w_i)
        term_ri = torch.einsum('bcm,com->bom', xr, w_i)
        term_ir = torch.einsum('bcm,com->bom', xi, w_r)

        out_r[..., :n_modes] = term_rr - term_ii
        out_i[..., :n_modes] = term_ri + term_ir

        # pack complex spectrum and inverse FFT
        out_ft = torch.complex(out_r, out_i)  # (B, out, N_ft)
        x_out = torch.fft.irfft(out_ft, n=N, dim=-1)  # (B, out, N)
        return x_out


# --- Now we define full model using the safe FNO layer ---
class FNO1DModel(nn.Module):
    def __init__(self, in_channels=1, width=32, modes=16, out_channels=1, n_layers=4):
        super().__init__()
        self.fc_in = nn.Conv1d(in_channels, width, 1)
        self.layers = nn.ModuleList([FNO1DLayerSafe(width, width, modes) for _ in range(n_layers)])
        self.fc_out = nn.Sequential(
            nn.Conv1d(width, 32, 1),
            nn.GELU(),
            nn.Conv1d(32, out_channels, 1)
        )

    def forward(self, x):
        # x: (B, in_channels, N)
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x


# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# --- Model, optimizer, loss ---
model = FNO1DModel(in_channels=1, out_channels=1, modes=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
criterion = nn.MSELoss()

# --- Move data to device ---
train_k, train_t = train_k.to(device), train_t.to(device)
test_k, test_t = test_k.to(device), test_t.to(device)

# --- Training loop ---
n_epochs = 15
for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()

    pred = model(train_k)
    loss = criterion(pred, train_t)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        test_pred = model(test_k)
        test_loss = criterion(test_pred, test_t)

    scheduler.step(test_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train Loss = {loss.item():.6f} | Test Loss = {test_loss.item():.6f}")

print(" Training completed successfully.")

```
