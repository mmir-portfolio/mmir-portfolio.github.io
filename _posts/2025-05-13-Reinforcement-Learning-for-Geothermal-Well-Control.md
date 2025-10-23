---
layout: post
title: Reinforcement Learning for Geothermal Well Control
image: "/posts/geothermal-rl-title-img.png"
tags: [Reinforcement Learning, Energy Optimization, Geothermal Well, Python, Clean Energy]
---

# Geothermal Energy Harvesting: A Calculation Framework

This project provides a **comprehensive framework** to understand how geothermal energy is harvested, calculated, and evaluated. It moves from first principles to real-world applications, blending theory, practical calculation, and modern computational techniques such as **reinforcement learning (RL)** for optimization of well control strategies.  

## Table of Contents
- [00. Introduction](#introduction-main)
- [01. Background on Geothermal Energy](#background-main)
- [02. Principles of Energy Harvesting](#principles-main)
- [03. Thermal Power Output Equation](#thermal-power-main)
- [04. Detailed Explanation of Variables](#variables-main)
- [05. Example Calculation](#example-main)
- [06. Conversion to Electricity](#conversion-main)
- [07. Annual Energy Yield](#annual-main)
- [08. Discussion of Results](#discussion-main)
- [09. Assumptions and Limitations](#assumptions-main)
- [10. Real-World Applications](#applications-main)
- [11. Future Directions](#future-main)
- [12. Conclusion](#conclusion-main)
- [13. Python Implementation](#python-main)

---

# 00. Introduction <a name="introduction-main"></a>

Geothermal energy is a **renewable, reliable, and base-load capable** energy source. Unlike solar and wind power, which fluctuate with weather and diurnal cycles, geothermal energy is available 24/7. This makes it one of the few renewable technologies capable of providing stable, dispatchable electricity for the grid.  



By the end of this document, you will:  

- Understand the **thermodynamic basis** for geothermal energy calculations.  
- See **step-by-step worked examples** of thermal power and electricity conversion.  
- Appreciate the **assumptions and constraints** behind these models.  
- Gain insight into **real-world geothermal projects**.  
- Learn how **Python and RL frameworks** can simulate and optimize geothermal systems.  

---

# 01. Background on Geothermal Energy <a name="background-main"></a>

The Earth’s crust stores immense amounts of thermal energy. This heat originates from:  

1. **Residual planetary heat** retained since Earth’s formation.  
2. **Radioactive decay** of isotopes such as uranium, thorium, and potassium.  
3. **Mantle convection**, which transfers heat toward the crust.  

The global geothermal gradient averages about **25–30 °C/km**, though in tectonically active regions, it can exceed **80 °C/km**. When permeable rock formations intersect with circulating groundwater, **geothermal reservoirs** form.  

Harnessing this energy involves drilling production wells to bring hot water or steam to the surface, extracting useful heat, and then reinjecting the cooled fluid through reinjection wells.  

---

# 02. Principles of Energy Harvesting <a name="principles-main"></a>

The fundamental principle is **thermal extraction from a circulating fluid**.  

- **Temperature difference (ΔT):** The bigger the gap between production temperature and reinjection temperature, the greater the available energy.  
- **Mass flow rate (ṁ):** Higher flow rates mean more thermal energy can be carried per unit time.  

Applications fall into two main categories:  

- **Direct use applications:** District heating, greenhouse heating, aquaculture, snow melting, and industrial drying. These have higher overall energy efficiency since no thermal-to-electricity conversion losses occur.  
- **Indirect use applications:** Electricity generation using steam turbines or Organic Rankine Cycle (ORC) systems. These are essential for integrating geothermal into power grids but have conversion efficiencies of **10–20%** depending on technology.  

---

# 03. Thermal Power Output Equation <a name="thermal-power-main"></a>

The basic thermal power equation is:  

<div style="text-align:center;">
Q = ṁ · c<sub>p</sub> · (T<sub>prod</sub> - T<sub>inj</sub>)
</div>

Where:  

- **Q** = thermal power output (W)  
- **ṁ** = mass flow rate (kg/s)  
- **c<sub>p</sub>** = specific heat capacity of water (4180 J/kg·K)  
- **T<sub>prod</sub>** = production temperature (°C)  
- **T<sub>inj</sub>** = reinjection temperature (°C)  

This equation assumes single-phase liquid water with no phase change. For high-enthalpy steam-dominated systems, enthalpy-based calculations are used instead.  

---

# 04. Detailed Explanation of Variables <a name="variables-main"></a>

1. **Mass flow rate (ṁ):**  
   Typical geothermal wells produce **20–100 kg/s**. This is influenced by reservoir permeability, pressure, and well design.  

2. **Specific heat capacity (c<sub>p</sub>):**  
   For liquid water, ~4180 J/kg·K at near-surface conditions. Slight variations occur at higher pressures and temperatures.  

3. **Production temperature (T<sub>prod</sub>):**  
   Reservoirs can vary widely:  
   - Low enthalpy: 50–90 °C  
   - Medium enthalpy: 90–150 °C  
   - High enthalpy: 150–300+ °C  

4. **Injection temperature (T<sub>inj</sub>):**  
   Reinjection typically occurs at **40–60 °C**, balancing heat extraction efficiency and reservoir sustainability.  

---

# 05. Example Calculation <a name="example-main"></a>

Let’s assume:  

- ṁ = 30 kg/s  
- c<sub>p</sub> = 4180 J/kg·K  
- T<sub>prod</sub> = 120 °C  
- T<sub>inj</sub> = 60 °C  

**Step 1: Thermal power (Q):**

<div style="text-align:center;">
Q = 30 · 4180 · (120 - 60) = 7.524 × 10<sup>6</sup> W ≈ 7.5 MW<sub>th</sub>
</div>

Thus, the geothermal well delivers **7.5 MW thermal power** continuously.  

---

# 06. Conversion to Electricity <a name="conversion-main"></a>

Assume **ORC efficiency = 12%**.  

<div style="text-align:center;">
P<sub>el</sub> = 0.12 · 7.5 = 0.9 MW<sub>el</sub>
</div>

So, the power plant produces **0.9 MW electricity**.  

---

# 07. Annual Energy Yield <a name="annual-main"></a>

Over one year:  

<div style="text-align:center;">
E = 0.9 · 8760 = 7884 MWh/year
</div>

This is enough to power roughly **700 Canadian households** annually, assuming average use of 11 MWh/year.  

---

# 08. Discussion of Results <a name="discussion-main"></a>

Key insights:  

- **ΔT dominates output:** Doubling ΔT roughly doubles thermal output.  
- **Flow rate scaling is linear:** If ṁ doubles, Q doubles.  
- **Efficiency bottleneck:** Conversion losses limit electrical output.  
- **System design trade-offs:** Higher reinjection temperatures protect the reservoir but reduce power output.  

---

# 09. Assumptions and Limitations <a name="assumptions-main"></a>

- No heat losses in pipes or pumps.  
- Constant fluid properties (c<sub>p</sub> = 4180 J/kg·K).  
- Efficiency fixed at 12%.  
- Long-term reservoir cooling not considered.  
- Reinjection assumed at fixed temperature regardless of seasonal changes.  

---

# 10. Real-World Applications <a name="applications-main"></a>

- **Iceland:** Over 90% of homes heated with geothermal.  
- **USA – The Geysers, California:** Largest geothermal field, >1500 MW.  
- **Kenya:** Rift Valley geothermal development surpassing 900 MW.  
- **Turkey:** Rapid ORC expansion (>1600 MW installed capacity by 2023).  

These examples demonstrate geothermal’s scalability and adaptability.  

---

# 11. Future Directions <a name="future-main"></a>

- **Enhanced Geothermal Systems (EGS):** Unlocking heat in impermeable rock via hydraulic stimulation.  
- **Hybrid plants:** Geothermal + solar thermal or biomass to raise efficiency.  
- **Carbon storage integration:** Combining geothermal reservoirs with CO₂ sequestration.  
- **Direct-use growth:** Agriculture, aquaculture, and industrial applications.  

---

# 12. Conclusion <a name="conclusion-main"></a>

- Geothermal is a **continuous renewable** energy source.  
- Even modest systems (~30 kg/s, ΔT = 60 °C) yield nearly **8 GWh annually**.  
- Wider adoption could reduce fossil fuel reliance and stabilize renewable-heavy grids.  

---

# 13. Python Implementation <a name="python-main"></a>

Below is a Python script for geothermal power calculations, followed by a prototype **reinforcement learning (RL) environment** for well control optimization.  

### A. Geothermal Calculation Functions

```python
import gym
import numpy as np
from stable_baselines3 import PPO

# --- Custom geothermal environment ---
class GeothermalEnv(gym.Env):
    def __init__(self):
        super(GeothermalEnv, self).__init__()
        # Action: flow control factor [0,1]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # Observation: [T_prod, T_inj, demand]
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.array([120.0, 60.0, 7.5e6])  # [T_prod °C, T_inj °C, demand W]
        return self.state

    def step(self, action):
        T_prod, T_inj, demand = self.state
        flow_factor = float(action[0])  # 0–1 scaling
        delta_T = T_prod - T_inj
        # Thermal output Q = ṁ * c_p * ΔT, assume ṁ=30 kg/s
        Q = flow_factor * 30 * 4180 * delta_T
        reward = -abs(Q - demand)  # penalize mismatch with demand
        done = False
        info = {"Q_output": Q, "Demand": demand}
        return self.state, reward, done, info

# --- Train PPO agent ---
env = GeothermalEnv()
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50_000)  # train for 50,000 steps

# Test trained agent
obs = env.reset()
print("Initial State:", obs)

for step in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Step {step+1}:")
    print(f"  Action (flow factor): {action[0]:.3f}")
    print(f"  Thermal Power Output (Q): {info['Q_output'] / 1e6:.2f} MW_th")
    print(f"  Demand: {info['Demand'] / 1e6:.2f} MW_th")
    print(f"  Reward: {reward:.2f}")
```






