# 03. Problem Formulation

The CO₂ injection process is formulated as a **sequential decision-making problem under uncertainty**, where control actions must balance competing physical and operational objectives.

At each time step \(k\), the system is characterized by an estimated state:

<div>
x^k = [ P^k , S^k ]
</div>

obtained via the Kalman filter.

Based on this estimate, the objective is to determine an optimal injection rate \(u_k\) that:

- Maximizes CO₂ storage (increase saturation \(S\))  
- Minimizes pressure buildup to avoid safety risks  
- Respects operational constraints on injection rates  

This leads to a constrained optimization problem:

<div>
min over u_k  J(u_k, x^k)
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

<div>
min over x in {0,1}^n   x^T Q x
</div>

where:

- \(x\): binary decision variables encoding discrete injection levels  
- \(Q\): problem-specific matrix capturing objective terms and penalties  

---

## 4.1 Decision Encoding

The injection rate \(u_k\) is discretized into binary variables:

<div>
u_k = sum(i=1 to n) w_i * x_i
</div>

where:
- \(w_i\) are predefined weights corresponding to injection levels  

---

## 4.2 Objective Construction

The QUBO matrix \(Q\) incorporates:

### Storage Reward Term  
Encourages higher saturation:

<div>
- λ_s * S^k
</div>

---

### Pressure Penalty Term  
Penalizes high pressure:

<div>
λ_p * (P^k / P_max)^2
</div>

---

### Control Regularization  
Prevents excessive injection:

<div>
λ_u * (u_k)^2
</div>

---

## 4.3 Final QUBO Structure

The combined formulation yields a quadratic objective:

<div>
J(x) = x^T Q x + c^T x
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

<div>
x^k
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
<div>
x_(k+1) = A x_k + B u_k + w_k
</div>

---

### Measurement Model
<div>
z_k = C x_k + v_k
</div>

---

### State Estimation
<div>
x^k = KF(z_k)
</div>

---

## 6.2 Data Collected

The simulation produces:

- True states:  
  <div>x_k</div>

- Estimated states:  
  <div>x^k</div>

- Measurements:  
  <div>z_k</div>

- Control actions:  
  <div>u_k</div>

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
