---
layout: post
title: Reinforcement Learning for Geothermal Well Control
image: "/posts/geothermal-rl-title-img.png"
tags: [Reinforcement Learning, Energy Optimization, Geothermal Well, Python, Clean Energy]
---
# Geothermal Energy Harvesting: A Calculation Framework

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

Geothermal energy is one of the most reliable forms of renewable energy. Unlike solar or wind, which depend heavily on weather conditions, geothermal power can provide **continuous base-load electricity** throughout the year. This makes it a crucial contributor to sustainable energy portfolios, particularly in regions with high heat flow or accessible geothermal reservoirs.  

The purpose of this document is to provide a **step-by-step framework for calculating geothermal thermal power output**, converting it to electricity, and estimating annual energy production. Beyond just equations, this guide includes explanations, examples, and considerations for real-world deployment.

By the end of this document, a reader should be able to understand how geothermal systems are evaluated quantitatively and how these calculations inform energy planning and feasibility studies.

---

# 01. Background on Geothermal Energy <a name="background-main"></a>

Geothermal energy exploits the natural heat stored beneath the Earth's surface. The primary sources of this heat include:

1. **Residual heat from planetary formation**  
2. **Radioactive decay**  
3. **Heat transfer from the Earth's core and mantle**

Geothermal energy is accessed by drilling wells into permeable rock formations containing hot water or steam. The amount of energy that can be harnessed depends on:

- **Temperature difference** between production and reinjection  
- **Flow rate** of geothermal fluid  
- **Specific heat capacity** of the working fluid

---

# 02. Principles of Energy Harvesting <a name="principles-main"></a>

The core principle is: **capture the thermal energy of fluids extracted from underground reservoirs and utilize it before reinjection.**  

Key points:

- The **temperature difference** between extracted and reinjected fluid determines energy captured  
- The **mass flow rate** of the fluid influences total energy extracted  

Uses:

- **Direct use:** Heating buildings, industrial processes, aquaculture, greenhouses  
- **Indirect use:** Electricity generation via turbines or ORC systems  

---

# 03. Thermal Power Output Equation <a name="thermal-power-main"></a>

The **thermal power** is:

<div style="text-align:center;">
Q = ṁ · c<sub>p</sub> · (T<sub>prod</sub> - T<sub>inj</sub>)
</div>

Where:

- <b>Q</b> = thermal power (W)  
- <b>ṁ</b> = mass flow rate (kg/s)  
- <b>c<sub>p</sub></b> = specific heat (J/kg·K)  
- <b>T<sub>prod</sub></b> = production temperature (°C)  
- <b>T<sub>inj</sub></b> = injection temperature (°C)  

---

# 04. Detailed Explanation of Variables <a name="variables-main"></a>

- **Mass flow rate (ṁ):** 20–100 kg/s typical  
- **Specific heat (c<sub>p</sub>):** ~4180 J/kg·K for water  
- **Production temperature (T<sub>prod</sub>):** Depends on reservoir depth and geothermal gradient  
- **Injection temperature (T<sub>inj</sub>):** Typically 40–60 °C  

---

# 05. Example Calculation <a name="example-main"></a>

We estimate geothermal thermal power output as:

\[
Q = \dot{m} \cdot c_p \cdot (T_{\rm prod} - T_{\rm inj})
\]

Where:

- \(\dot{m}\) : mass flow rate (kg/s)  
- \(c_p\) : specific heat of water (~4180 J/kg·K)  
- \(T_{\rm prod}, T_{\rm inj}\) : production and injection temperatures (°C)  

---

## Example Calculation

Suppose:

- \(\dot{m} = 30 \, \text{kg/s}\)  
- \(c_p = 4180 \, \text{J/kg·K}\)  
- \(T_{\rm prod} = 120^\circ \text{C}\)  
- \(T_{\rm inj} = 60^\circ \text{C}\)  

Then the thermal power is:

\[
Q = 30 \cdot 4180 \cdot (120 - 60)
\]

\[
Q = 7.524 \times 10^6 \, \text{W} = 7.5 \, \text{MW}_{\rm th}
\]

If converted to electricity with 12% efficiency:

\[
P_{\rm el} = 0.12 \cdot 7.5 = 0.9 \, \text{MW}_{\rm el}
\]

Over a year (8760 hours):

\[
E = 0.9 \cdot 8760 = 7884 \, \text{MWh/year}
\]


---

# 07. Annual Energy Yield <a name="annual-main"></a>

<div style="text-align:center;">
E = P<sub>el</sub> · 8760  
E = 0.9 · 8760  
E ≈ 7884 MWh/year
</div>

Enough for ~700 Canadian households (~11 MWh/year each).

---

# 08. Discussion of Results <a name="discussion-main"></a>

- **Temperature difference matters**  
- **Flow rate matters**  
- **Efficiency is limited**  
- **Scalability possible with multiple wells**  

---

# 09. Assumptions and Limitations <a name="assumptions-main"></a>

- Working fluid = water, no phase change  
- No pipe or pump losses considered  
- Efficiency fixed at 12%  
- Reservoir sustainability not considered  

---

# 10. Real-World Applications <a name="applications-main"></a>

- **Iceland:** ~90% households geothermal heating  
- **USA (The Geysers):** >1500 MW  
- **Kenya (Rift Valley):** >900 MW  
- **Turkey:** Rapid ORC plant expansion  

---

# 11. Future Directions <a name="future-main"></a>

- **EGS:** Fracturing rocks for deeper heat  
- **Hybrid systems:** Geothermal + solar/biomass  
- **Carbon capture synergy**  
- **Direct-use expansion**  

---

# 12. Conclusion <a name="conclusion-main"></a>

- Reliable base-load power  
- Modest systems: ~8 GWh/year  
- Wider adoption reduces fossil fuel reliance and stabilizes grids  

---

# 13. Python Implementation <a name="python-main"></a>

Below is a simplified prototype implementation using **Stable Baselines3 (PPO)**:

```
import gym
import numpy as np
from stable_baselines3 import PPO

# Custom geothermal environment
class GeothermalEnv(gym.Env):
    def __init__(self):
        super(GeothermalEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32)
        self.state = np.array([100.0, 60.0, 50.0])  # [T_prod, T_inj, demand]

    def reset(self):
        self.state = np.array([100.0, 60.0, 50.0])
        return self.state

    def step(self, action):
        T_prod, T_inj, demand = self.state
        flow_factor = action[0]
        delta_T = T_prod - T_inj
        Q = flow_factor * 4180 * delta_T  # simplified thermal output
        reward = -abs(Q - demand)  # penalize mismatch with demand
        self.state = np.array([T_prod, T_inj, demand])  # static for demo
        done = False
        return self.state, reward, done, {}

# Train RL agent
env = GeothermalEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)  # train for 50,000 steps
```

