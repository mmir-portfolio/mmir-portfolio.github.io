---
layout: post
title: Quantum-Enhanced Digital Twin for CO2 Storage Optimization (Kalman Filtering and QAOA-Based Injection Optimization)
image: "/posts/quantum-enhanced-digital-twin-img.png"
tags: [Quantum Computing, Digital Twin, Carbon Storage, Optimization, Kalman Filter, Decision-making]
---

In this project, we developed Quantum-enhanced Digital Twin for CO2 Storage Optimization that integrates Kalman Filter-based state estimation and Quantum Optimization employing QAOA. This is a closed-loop decision-making system, which can optimize injection strategies leveraging estimated reservoir states.

## Table of Contents
- [00. Introduction](#introduction-main)
- [01. Background](#background-main)
- [02. Governing Physics](#physics-main)
- [03. Problem Formulation](#problem-main)
- [04. Quantum Formulation (QUBO)](#qubo-main)
- [05. Hybrid Workflow](#workflow-main)
- [06. Dataset Generation & Simulation](#dataset-main)
- [07. Industry Relevance](#industry-main)
- [08. Python Implementation (Core Concept)](#python-main)

---

# 00. Introduction <a name="introduction-main"></a>

Carbon Capture and Storage (CCS) requires precise control of CO₂ injection to maximize storage efficiency while preventing pressure-induced leakage and long-term reservoir damage. This is a challenging problem due to:

- Nonlinear subsurface dynamics  
- Operational constraints  
- Limited and noisy sensor observations  

Traditional optimization approaches struggle under uncertainty and lack real-time adaptability.

This project introduces a **hybrid quantum-classical digital twin framework** that integrates:

- A physics-informed digital twin  
- Kalman filter–based state estimation  
- Quantum optimization using QAOA  

Together, these form a **closed-loop decision-making system**, continuously optimizing injection strategies using estimated reservoir states.

---

# 01. Background <a name="background-main"></a>

Carbon Capture and Storage (CCS) is essential for reducing industrial CO₂ emissions while sustaining energy production.

### Key Challenges

- Pressure buildup leading to leakage risks  
- Limited observability of subsurface states  
- Complex nonlinear dynamics  
- High-dimensional optimization problems  

### Key Technologies Used

- **Digital Twin:** Real-time system representation  
- **Kalman Filtering:** State estimation under uncertainty  
- **QAOA:** Quantum optimization via QUBO formulation  

This project combines these into a unified framework for **adaptive, intelligent decision-making**.

---
# 02. Governing Physics<a name="physics-main"></a>

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
    <br><br> 0         1 − δΔt ]
<br><br>
B = [ βΔt  
      γΔt ]
<br><br>
C = [ 1   0 ]
</div>

Key observations:

- Only **pressure (P)** is directly observed, reflecting realistic sensing limitations  
- **Saturation (S)** is treated as a hidden state and must be estimated using the Kalman filter  

# 03. Problem Formulation <a name="problem-main"></a>

The CO₂ injection process is formulated as a **sequential decision-making problem under uncertainty**, where control actions must balance competing physical and operational objectives.

At each time step k, the system is characterized by an estimated state:

<div style="text-align:center;">
x<sup>k</sup> = [ P<sup>k</sup> , S<sup>k</sup> ]
</div>

obtained via the Kalman filter. Based on this estimate, the objective is to determine an optimal injection rate u<sub>k</sub> that:

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

# 04. Quantum Formulation (QUBO) <a name="qubo-main"></a>

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

The QUBO matrix <em>Q</em> is constructed to encode the key physical objectives governing CO₂ injection, translating reservoir dynamics and operational constraints into an optimization framework.

The objective consists of three main components:

<ul>
<li><strong>Storage Reward Term</strong>, encouraging higher CO₂ saturation</li>
<li><strong>Pressure Penalty Term</strong>, discouraging unsafe pressure buildup</li>
<li><strong>Control Regularization</strong>, limiting excessive injection rates</li>
</ul>

---

### 4.2.1 Storage Reward Term

This term promotes efficient CO₂ storage by favoring higher saturation levels:

<div style="text-align:center;">
−λ<sub>s</sub> S<sub>k</sub>
</div>

In the QUBO formulation, this is implemented through negative linear coefficients, encouraging the selection of injection actions that increase saturation.

---

### 4.2.2 Pressure Penalty Term

To ensure safe operation, pressure buildup is penalized using a smooth quadratic function:

<div style="text-align:center;">
λ<sub>p</sub> (P<sub>k</sub> / P<sub>max</sub>)<sup>2</sup>
</div>

This formulation ensures that the penalty is active across all operating conditions, increasing progressively as pressure approaches critical limits. The term is incorporated into the linear coefficients of the QUBO, influencing all decision variables.

---

### 4.2.3 Control Regularization

To prevent overly aggressive injection strategies, a regularization term is introduced:

<div style="text-align:center;">
λ<sub>u</sub> (u<sub>k</sub>)<sup>2</sup>
</div>

where the control input is defined as:

<div style="text-align:center;">
u<sub>k</sub> = Σ w<sub>i</sub> x<sub>i</sub>
</div>

Expanding this term:

<div style="text-align:center;">
(u<sub>k</sub>)<sup>2</sup> = Σ w<sub>i</sub><sup>2</sup> x<sub>i</sub> + Σ w<sub>i</sub> w<sub>j</sub> x<sub>i</sub> x<sub>j</sub>
</div>

This results in:

<ul>
<li><strong>Diagonal terms</strong> in the QUBO matrix, penalizing individual injection decisions</li>
<li><strong>Off-diagonal terms</strong>, introducing coupling between decision variables</li>
</ul>

---

### Combined Objective Representation

These components are combined to form a unified objective that balances storage efficiency, operational safety, and control stability. The resulting QUBO formulation enables structured optimization of injection strategies under dynamic reservoir conditions.

---

## 4.3 Final QUBO Structure

The combined formulation yields a quadratic objective:

<div style="text-align:center;">
J(x) = x<sup>T</sup> Q x + c<sup>T</sup> x
</div>

where the linear term <em>c</em> and quadratic matrix <em>Q</em> encode the underlying physical objectives of the system, including:

<ul>
<li><strong>Storage reward</strong>, encouraging higher CO₂ saturation</li>
<li><strong>Pressure penalty</strong>, discouraging unsafe pressure buildup</li>
<li><strong>Control regularization</strong>, limiting excessive injection rates</li>
</ul>

The control input is defined as:

<div style="text-align:center;">
u<sub>k</sub> = Σ w<sub>i</sub> x<sub>i</sub>
</div>

where <em>w<sub>i</sub></em> represent discretized injection levels corresponding to physically meaningful control actions.

Substituting this into the objective function, the optimization problem becomes:

<div style="text-align:center;">
J(x) = −λ<sub>s</sub> S<sub>k</sub> (Σ w<sub>i</sub> x<sub>i</sub>)
+ λ<sub>p</sub> (P<sub>k</sub> / P<sub>max</sub>)<sup>2</sup> (Σ w<sub>i</sub> x<sub>i</sub>)
+ λ<sub>u</sub> (Σ w<sub>i</sub> x<sub>i</sub>)<sup>2</sup>
</div>

This formulation ensures that each binary decision variable contributes proportionally to the objective based on its associated injection magnitude.

Expanding the quadratic term:

<div style="text-align:center;">
(Σ w<sub>i</sub> x<sub>i</sub>)<sup>2</sup> =
Σ w<sub>i</sub><sup>2</sup> x<sub>i</sub> +
Σ w<sub>i</sub> w<sub>j</sub> x<sub>i</sub> x<sub>j</sub>
</div>

results in:

<ul>
<li><strong>Diagonal terms</strong> in the QUBO matrix, representing the cost of activating individual injection levels, scaled by their physical magnitude</li>
<li><strong>Off-diagonal terms</strong>, capturing interactions between different injection levels through pairwise coupling</li>
</ul>

This weighted formulation ensures that the optimization reflects the true physical impact of each control action, rather than treating all binary decisions equally.

This QUBO problem is solved using the <strong>Quantum Approximate Optimization Algorithm (QAOA)</strong>, which leverages parameterized quantum circuits to approximate the optimal binary solution. The QUBO coefficients are dynamically updated at each time step using the estimated state from the digital twin, enabling adaptive and closed-loop decision-making.

---

# 05. Hybrid Workflow <a name="workflow-main"></a>

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

This loop continues iteratively, enabling **adaptive control under uncertainty**.

---

# 06. Dataset Generation & Simulation <a name="dataset-main"></a>

Since real reservoir data is not publicly available, a **synthetic dataset** is generated using the physics-based model augmented with stochastic noise.

---

## 6.1 Simulation Process

At each time step:

### 1. The true system evolves according to:
<div style="text-align:center;">
x<sub>k+1</sub> = A x<sub>k</sub> + B u<sub>k</sub> + w<sub>k</sub>
</div>

---

### 2. Measurements are generated:
<div style="text-align:center;">
z<sub>k</sub> = C x<sub>k</sub> + v<sub>k</sub>
</div>

---

### 3. The Kalman filter estimates the state:
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

# 07. Industry Relevance <a name="industry-main"></a>

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
  
 ---

# 08. Python Implementation (Core Concept) <a name="python-main"></a>

This section implements the hybrid quantum-classical Digital Twin for CCS. The architecture integrates:

- Digital Twin (physics-based model)  
- Kalman Filter (state estimation)  
- QUBO formulation (optimization encoding)  
- QAOA (quantum optimization)  
- Closed-loop control system  


```python
# === 8.1. ENVIRONMENT SETUP ===
# !pip install --upgrade pip setuptools wheel
# !pip uninstall -y qiskit qiskit-optimization qiskit-algorithms
# !pip install qiskit==0.45.0 qiskit-optimization==0.6.0 qiskit-algorithms==0.2.1
# !pip install qiskit-optimization
# !pip install qiskit_algorithms

# === 8.2. IMPORTS ===
import numpy as np
import matplotlib.pyplot as plt

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# === 8.3. QUBO CONSTRUCTION ===
def build_quadratic_program(state):
    pressure, saturation = state
    qp = QuadraticProgram()

    # Hyperparameters (tunable)
    lambda_s = 1.0   # storage reward weight
    lambda_p = 5.0   # pressure penalty weight
    lambda_u = 0.5   # control regularization weight
    P_max = 5.0

    # Injection weights (define discrete levels)
    weights = [1.0, 0.7, 0.3]

    # Binary decision variables
    for i in range(len(weights)):
        qp.binary_var(name=f"x{i}")

    # --- Compute pressure penalty (smooth) ---
    pressure_penalty = lambda_p * (pressure / P_max) ** 2

    # --- Linear terms ---
    linear = {}
    for i in range(len(weights)):
          linear[f"x{i}"] = (
              -lambda_s * saturation * weights[i]     # Storage reward (scaled)
              + pressure_penalty * weights[i]         # Pressure penalty (scaled)
              + lambda_u * weights[i]**2              # Control regularization (diagonal)
          )

    # --- Quadratic terms (control regularization expansion) ---
    quadratic = {}
    for i in range(len(weights)):
       for j in range(i+1, len(weights)):
           quadratic[(f"x{i}", f"x{j}")] = lambda_u * weights[i] * weights[j]

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp

# === 8.4. QAOA SOLVER ===
def solve_with_qaoa(qp):
    sampler = StatevectorSampler()
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=50),
        reps=1
    )
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    solution = np.array(result.x)
    return solution, result.fval

# === 8.5. ACTION DECODING ===
def decode_solution(x):
    # Map binary solution to continuous injection rate
    return 0.5 * x[0] + 1.0 * x[1] + 2.0 * x[2]

# === 8.6. QUANTUM DECISION STEP ===
def quantum_qaoa_step(simulator, state):
    qp = build_quadratic_program(state)
    solution, cost = solve_with_qaoa(qp)
    injection_rate = decode_solution(solution)
    next_state = simulator.step(state, injection_rate)
    return next_state, injection_rate, solution

# === 8.7. DIGITAL TWIN + KALMAN FILTER ===
class KalmanDigitalTwin:
    def __init__(self, dt=0.1):
        self.dt = dt
        alpha, beta = 0.05, 0.2
        gamma, delta = 0.1, 0.05

        # State-space matrices
        self.A = np.array([
            [1 - alpha*dt, 0],
            [0, 1 - delta*dt]
        ])
        self.B = np.array([
            [beta*dt],
            [gamma*dt]
        ])
        self.C = np.array([[1.0, 0.0]])

        # Process and measurement noise
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.05]])

        # Initial state estimate
        self.mu = np.array([1.0, 0.1])
        self.Sigma = np.eye(2)

    def predict(self, u):
        self.mu = self.A @ self.mu + self.B.flatten() * u
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.Q

    def update(self, z):
        K = self.Sigma @ self.C.T @ np.linalg.inv(
            self.C @ self.Sigma @ self.C.T + self.R
        )
        self.mu = self.mu + K @ (z - self.C @ self.mu)
        self.Sigma = (np.eye(2) - K @ self.C) @ self.Sigma

    def step(self, true_state, u):
        process_noise = np.random.multivariate_normal([0, 0], self.Q)
        true_state = self.A @ true_state + self.B.flatten() * u + process_noise
        measurement_noise = np.random.normal(0, np.sqrt(self.R[0,0]))
        z = self.C @ true_state + measurement_noise
        self.predict(u)
        self.update(z)
        return true_state, self.mu, z

# === 8.8. CLOSED-LOOP SIMULATION ===
twin = KalmanDigitalTwin()
true_state = np.array([1.0, 0.1])
states_true = []
states_est = []
actions = []

for t in range(20):
    est_state = twin.mu
    est_state, injection_rate, _ = quantum_qaoa_step(twin, est_state)
    true_state, estimated_state, measurement = twin.step(true_state, injection_rate)

    states_true.append(true_state)
    states_est.append(estimated_state)
    actions.append(injection_rate)

    print(f"Step {t}:")
    print(f"  True State: {true_state}")
    print(f"  Estimated State: {estimated_state}")
    print(f"  Measurement: {measurement}")
    print(f"  Injection: {injection_rate}")

# === 8.9. VISUALIZATION ===
states_true = np.array(states_true)
states_est = np.array(states_est)

plt.figure()
plt.plot(states_true[:,0], label="True Pressure")
plt.plot(states_est[:,0], '--', label="Estimated Pressure")
plt.legend()
plt.title("Kalman Filter State Estimation")
plt.grid()
plt.show()

```
