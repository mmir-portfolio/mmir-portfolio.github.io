# 02. Governing Physics

The physical behavior of CO₂ injection in a subsurface reservoir is governed by complex multiphase flow dynamics, including:

- Pressure diffusion  
- Fluid transport  
- Interactions with porous geological media  

While high-fidelity models involve **partial differential equations (PDEs)**, this project adopts a **reduced-order, physics-informed dynamical system** to enable efficient simulation and integration with optimization algorithms.

---

## 2.1 State Variables

The system is described using a compact state vector:

<div style="text-align:center; font-size:18px;">
x = 
<span style="display:inline-block; border-left:2px solid black; border-right:2px solid black; padding:6px 10px;">
P<br>
S
</span>
</div>

Where:

- P = reservoir pressure  
- S = CO₂ saturation  

---

## 2.2 Dynamic Model

The evolution of the system is approximated using linearized dynamics:

<div style="text-align:center;">
dP/dt = − α P + β u  
<br>
dS/dt = γ u − δ S
</div>

Where:

- u = CO₂ injection rate (control input)  
- α = pressure dissipation coefficient  
- β = pressure response to injection  
- γ = saturation increase due to injection  
- δ = saturation decay or redistribution  

These equations capture the essential trade-off:

- Increasing injection improves storage (higher saturation)  
- But also increases pressure, introducing operational risk  

---

## 2.3 Discrete-Time State-Space Formulation

For integration with estimation and control, the system is discretized:

<div style="text-align:center;">
x<sub>k+1</sub> = A x<sub>k</sub> + B u<sub>k</sub> + w<sub>k</sub>  
<br>
z<sub>k</sub> = C x<sub>k</sub> + v<sub>k</sub>
</div>

Where:

- x<sub>k</sub> = system state at time step k  
- z<sub>k</sub> = measurement vector  
- w<sub>k</sub> ∼ N(0, Q) = process noise  
- v<sub>k</sub> ∼ N(0, R) = measurement noise  

---

## 2.4 System Matrices

<div style="text-align:center;">
A = [ 1 − αΔt   0  
      0         1 − δΔt ]
<br><br>
B = [ βΔt  
      γΔt ]
<br><br>
C = [ 1   0 ]
</div>

Key observations:

- Only **pressure (P)** is directly observed, reflecting realistic sensing limitations  
- **Saturation (S)** is treated as a hidden state and must be estimated using the Kalman filter  

# 03. Problem Formulation

The CO₂ injection process is formulated as a **sequential decision-making problem under uncertainty**, where control actions must balance competing physical and operational objectives.

At each time step k, the system is characterized by an estimated state:

<div style="text-align:center;">
x<sup>k</sup> = [ P<sup>k</sup> , S<sup>k</sup> ]
</div>

obtained via the Kalman filter.

Based on this estimate, the objective is to determine an optimal injection rate u<sub>k</sub> that:

- Maximizes CO₂ storage (increase saturation S)  
- Minimizes pressure buildup to avoid safety risks  
- Respects operational constraints on injection rates  

This leads to a constrained optimization problem:

<div style="text-align:center;">
min<sub>u<sub>k</sub></sub> J(u<sub>k</sub>, x<sup>k</sup>)
</div>

where the objective function encodes trade-offs between storage efficiency and pressure safety.

The optimization is performed in a **receding horizon manner**, where decisions are updated at each time step using the latest estimated state.

Due to:
- Combinatorial nature of discretized control actions  
- Nonlinear dependencies on system state  

the problem is reformulated into a **Quadratic Unconstrained Binary Optimization (QUBO)** problem, enabling quantum optimization techniques.

---

# 04. Quantum Formulation (QUBO)

To enable quantum optimization, the control problem is expressed in QUBO form:

<div style="text-align:center;">
min<sub>x ∈ {0,1}<sup>n</sup></sub> x<sup>T</sup> Q x
</div>

where:

- x represents binary decision variables encoding discrete injection levels  
- Q is a problem-specific matrix capturing objective terms and penalties  

---

## 4.1 Decision Encoding

The injection rate u<sub>k</sub> is discretized into binary variables:

<div style="text-align:center;">
u<sub>k</sub> = ∑<sub>i=1</sub><sup>n</sup> w<sub>i</sub> x<sub>i</sub>
</div>

where:
- w<sub>i</sub> are predefined weights corresponding to injection levels  

---

## 4.2 Objective Construction

The QUBO matrix Q incorporates:

### Storage Reward Term  
Encourages higher saturation:

<div style="text-align:center;">
− λ<sub>s</sub> S<sup>k</sup>
</div>

---

### Pressure Penalty Term  
Penalizes high pressure:

<div style="text-align:center;">
λ<sub>p</sub> (P<sup>k</sup> / P<sub>max</sub>)<sup>2</sup>
</div>

---

### Control Regularization  
Prevents excessive injection:

<div style="text-align:center;">
λ<sub>u</sub> (u<sub>k</sub>)<sup>2</sup>
</div>

---

## 4.3 Final QUBO Structure

The combined formulation yields a quadratic objective:

<div style="text-align:center;">
J(x) = x<sup>T</sup> Q x + c<sup>T</sup> x
</div>

This QUBO problem is solved using the **Quantum Approximate Optimization Algorithm (QAOA)**, which leverages parameterized quantum circuits to approximate the optimal binary solution.

---

# 05. Hybrid Workflow

The system operates as a **closed-loop hybrid quantum-classical pipeline**, integrating:

- State estimation  
- Physics simulation  
- Optimization  

---

## Workflow Steps

### 1. State Estimation  
Sensor measurements are processed by the Kalman filter to produce an estimated state:

<div style="text-align:center;">
x<sup>k</sup>
</div>

---

### 2. Problem Encoding  
The estimated state is used to construct a QUBO problem reflecting current system conditions.

---

### 3. Quantum Optimization  
QAOA is applied to solve the QUBO and determine the optimal injection decision.

---

### 4. System Update  
The chosen injection rate is applied to the physical model (digital twin).

---

### 5. Measurement Update  
New measurements are generated and fed back into the Kalman filter.

---

This loop continues iteratively, enabling **adaptive control under uncertainty**.

---

# 06. Dataset Generation & Simulation

Since real reservoir data is not publicly available, a **synthetic dataset** is generated using the physics-based model augmented with stochastic noise.

---

## 6.1 Simulation Process

At each time step:

### System Evolution
<div style="text-align:center;">
x<sub>k+1</sub> = A x<sub>k</sub> + B u<sub>k</sub> + w<sub>k</sub>
</div>

---

### Measurement Model
<div style="text-align:center;">
z<sub>k</sub> = C x<sub>k</sub> + v<sub>k</sub>
</div>

---

### State Estimation
<div style="text-align:center;">
x<sup>k</sup> = KF(z<sub>k</sub>)
</div>

---

## 6.2 Data Collected

The simulation produces:

- True states:  
  <div style="text-align:center;">x<sub>k</sub></div>

- Estimated states:  
  <div style="text-align:center;">x<sup>k</sup></div>

- Measurements:  
  <div style="text-align:center;">z<sub>k</sub></div>

- Control actions:  
  <div style="text-align:center;">u<sub>k</sub></div>

---

## 6.3 Purpose

This dataset supports:

- Model validation  
- Performance evaluation of estimation accuracy  
- Analysis of control policies  

---

# 07. Industry Relevance

This project reflects key challenges and solutions relevant to modern energy systems and industrial operations.

---

## 7.1 Digital Twin Integration

- Combines physics-based models with real-time data  
- Enables predictive monitoring and control  

---

## 7.2 Decision-Making Under Uncertainty

- Incorporates state estimation using Kalman filtering  
- Handles noisy and incomplete measurements  

---

## 7.3 Advanced Optimization

- Applies QAOA to structured industrial optimization problems  
- Explores quantum computing for complex decision-making  

---

## 7.4 Applicability

The framework is relevant to:

- Carbon Capture and Storage (CCS)  
- Reservoir management  
- Energy system optimization  
- Smart industrial control systems  
