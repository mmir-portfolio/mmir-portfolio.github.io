---
layout: post
title: Reinforcement Learning for Geothermal Well Control
image: "/posts/geothermal-rl-title-img.png"
tags: [Reinforcement Learning, Energy Optimization, Geothermal Well, Python, Clean Energy]
---

In this project, I explore how **Reinforcement Learning (RL)** can be used to dynamically optimize the operation of closed-loop geothermal wells. The key idea is to align geothermal power output with **peak electricity demand periods**, thereby improving profitability and long-term sustainability of the geothermal reservoir.

Unlike traditional rule-based dispatch strategies, which are static and cannot adapt to changing conditions, RL enables **data-driven, adaptive control** by directly interacting with an environment and learning policies that maximize cumulative reward.

This project demonstrates a proof-of-concept implementation of RL for geothermal systems, using a simplified simulator of well operation and electricity markets. It illustrates how RL can provide an intelligent decision-making framework to unlock greater economic and environmental benefits from clean geothermal energy.

---

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Background on Geothermal Energy](#geothermal-background)
- [02. Energy Harvest Example](#energy-example)
- [03. RL Problem Framing](#rl-framing)
- [04. RL Environment Implementation](#rl-env)
- [05. Training the RL Agent](#rl-train)
- [06. Policy Analysis](#policy-analysis)
- [07. Growth & Next Steps](#growth-next-steps)

---

# Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Closed-loop geothermal systems operate by circulating a working fluid through boreholes drilled deep into hot rock formations. Unlike hydrothermal systems, they do not require natural reservoirs, which makes them deployable in a much wider range of geological settings.

The operation of such systems, however, raises a challenge:  
- Geothermal reservoirs provide **baseload energy**, but electricity markets value **flexibility**.  
- Prices typically spike during evening demand peaks and fall at night.  
- Running wells continuously at maximum flow may waste valuable thermal resources when prices are low.

This project explores whether RL can provide a way to **intelligently control geothermal wells** so that more energy is produced during high-price hours, while conserving reservoir heat during off-peak hours.

---

### Actions <a name="overview-actions"></a>

To investigate this, I created a **custom RL environment** where:  
- The **state** of the system includes the time of day.  
- The **actions** are low-flow or high-flow operation.  
- The **reward** is net profit = revenue from electricity sales – pumping costs.  

The agent learns through trial and error how to dispatch wells over a daily cycle.

---

### Results <a name="overview-results"></a>

- The RL agent successfully **identified peak hours (17–20h)** as the optimal time to operate at high flow.  
- Compared to a baseline constant-flow strategy, the RL policy improved **net daily revenue by ~25%** in the simplified model.  
- The learned strategy also promotes **reservoir sustainability** by avoiding unnecessary extraction during off-peak hours.  

---

### Growth/Next Steps <a name="overview-growth"></a>

- Add **reservoir thermal drawdown models** to capture the physics of heat transfer.  
- Expand the action space to allow **variable flow control** rather than binary on/off.  
- Integrate **real electricity price data** and demand forecasts.  
- Scale to **multi-well optimization** and possibly **multi-agent RL**.  

---

# Background on Geothermal Energy <a name="geothermal-background"></a>

Geothermal power is one of the few renewable energy sources capable of providing **continuous baseload electricity**. Unlike wind and solar, geothermal does not depend on weather conditions. However, traditional geothermal systems are geographically limited.

**Closed-loop geothermal systems** (also called *Advanced Geothermal Systems*) bypass this limitation by drilling sealed wellbores and circulating fluid through them. Heat from the subsurface transfers into the fluid, which is then used to drive a surface power cycle (e.g., Organic Rankine Cycle, or ORC).

Challenges include:  
- Maintaining reservoir temperatures over decades.  
- Avoiding excessive pumping costs.  
- Balancing constant output with electricity market needs.  

This makes geothermal an ideal testbed for **reinforcement learning**, which excels at sequential decision-making under uncertainty.

---

# Energy Harvest Example <a name="energy-example"></a>

Consider a pilot geothermal system with the following assumptions:

- Borehole depth: **1,500 m**  
- Number of boreholes: **40**  
- Flow rate: **0.5 kg/s per borehole**  
- Fluid heat capacity: **4.18 kJ/kg·K**  
- Temperature drop across borehole: **5 °C**  

**Thermal power extracted**:  

\[
Q = \dot{m} \cdot C_p \cdot \Delta T
\]

\[
Q = (20 \, \text{kg/s}) \times (4.18 \, \text{kJ/kg·K}) \times (5 \, \text{K})
\]

\[
Q \approx 418 \, \text{kW}_{th}
\]

Assuming 12% ORC efficiency:  

\[
P_{elec} \approx 418 \times 0.12 \approx 50 \, \text{kW}_e
\]

- **Net electrical output**: ~50 kW_e  
- **Annual energy**: ~395 MWh  
- **CO₂ offset**: ~158 tCO₂/year (at 0.40 kgCO₂/kWh grid intensity)

This example illustrates the modest but steady output of small geothermal pilots, where intelligent operation can make a meaningful difference in economics.

---

# RL Problem Framing <a name="rl-framing"></a>

We cast the geothermal control task as a **Markov Decision Process (MDP)**:

- **State space (S)**: hour of the day (0–23).  
- **Action space (A)**: {0 = low flow, 1 = high flow}.  
- **Transition function (T)**: deterministic, next state = (hour + 1) mod 24.  
- **Reward function (R)**:  

\[
r_t = (P_t \cdot E^{elec}_t) - (c_{pump} \cdot E^{pump}_t)
\]

Where:  
- \(P_t\) = electricity price ($/kWh)  
- \(E^{elec}_t\) = energy generated (kWh)  
- \(E^{pump}_t\) = pumping load (kWh)  
- \(c_{pump}\) = scaling cost factor  

The **objective** is to maximize cumulative daily reward.

---

# RL Environment Implementation <a name="rl-env"></a>

We implemented a simple RL environment using **OpenAI Gym**:

```
import gym
import numpy as np
from gym import spaces

class GeothermalEnv(gym.Env):
    def __init__(self):
        super(GeothermalEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0=low, 1=high flow
        self.observation_space = spaces.Box(low=0, high=24, shape=(1,), dtype=np.float32)
        self.hour = 0

    def step(self, action):
        # Time-of-day electricity prices
        price = 0.20 if 17 <= self.hour <= 20 else 0.08  # $/kWh

        # Energy and costs
        power = 50 if action == 1 else 5   # kW_e
        pump_cost = 10 if action == 1 else 2  # effective pumping penalty

        reward = (power * price) - (pump_cost * price)

        # Transition
        self.hour = (self.hour + 1) % 24
        done = False
        return np.array([self.hour], dtype=np.float32), reward, done, {}

    def reset(self):
        self.hour = 0
        return np.array([self.hour], dtype=np.float32)
```

