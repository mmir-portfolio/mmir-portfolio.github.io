---
layout: post
title: Reinforcement Learning for Geothermal Well Control
image: "/posts/geothermal-rl-title-img.png"
tags: [Reinforcement Learning, Energy Optimization, Geothermal Well, Python, Clean Energy]
---

- [00. Project Overview](#overview-main)  
- [01. Introduction to Geothermal Systems](#intro-main)  
- [02. Why Closed-Loop Geothermal?](#closedloop-main)  
- [03. Energy Harvested Calculation](#energy-main)  
- [04. Reinforcement Learning Approach](#rl-main)  
- [05. Python Implementation](#python-main)  
- [06. Scaling to Multi-Well Fields](#scaling-main)  
- [07. Real-World Challenges](#challenges-main)  
- [08. Future Directions](#future-main)  
- [09. Results and Insights](#results-main)  
- [10. Conclusion](#conclusion-main)  

---

# Project Overview <a name="overview-main"></a>

This project demonstrates how **Reinforcement Learning (RL)** can be applied to closed-loop geothermal wells to optimize energy extraction.  
The aim is to dynamically balance **thermal output** with **grid demand**, while ensuring **long-term reservoir sustainability**.  

The study includes:  
- A thermodynamic model for estimating harvested power  
- An RL-based control framework  
- Simulation of well operation against variable demand  
- Insights into challenges and future potential  

---

# Introduction to Geothermal Systems <a name="intro-main"></a>

Geothermal energy taps into the Earth’s internal heat to produce electricity or direct-use heating.  
Conventional systems rely on hydrothermal resources, where naturally occurring hot water or steam is brought to the surface.  

Key advantages:  
- **Base-load power**: provides stable, around-the-clock generation  
- **Low emissions**: nearly carbon-free  
- **High capacity factor** compared to wind or solar  

Challenges:  
- Requires suitable geology (volcanic or tectonically active regions)  
- Risk of resource depletion if mismanaged  
- Induced seismicity in some enhanced geothermal projects  

---

# Why Closed-Loop Geothermal? <a name="closedloop-main"></a>

Closed-loop geothermal systems circulate a working fluid in **sealed wells**, preventing direct contact with reservoir fluids.  

Benefits:  
- Avoids scaling, corrosion, and brine handling issues  
- Can operate in a wider range of geologies  
- Improved control over injection/production conditions  

These systems are highly **engineerable**, making them a strong candidate for coupling with **smart control strategies** like RL.

---

# Energy Harvested Calculation <a name="energy-main"></a>

We estimate geothermal thermal power output as:

$$
Q = \dot{m} \cdot c_p \cdot (T_{prod} - T_{inj})
$$

Where:  
- $\dot{m}$ : mass flow rate (kg/s)  
- $c_p$ : specific heat of water (~4180 J/kg·K)  
- $T_{prod}, T_{inj}$ : production and injection temperatures (°C)  

---

### Example Calculation

Suppose:  

- $\dot{m} = 30 \, \text{kg/s}$  
- $c_p = 4180 \, \text{J/kg·K}$  
- $T_{prod} = 120^\circ \text{C}$  
- $T_{inj} = 60^\circ \text{C}$  

Then:

$$
Q = 30 \cdot 4180 \cdot (120 - 60)
$$

$$
Q = 7.524 \times 10^6 \, \text{W} = 7.5 \, \text{MW}_{th}
$$

If converted to electricity with 12% efficiency:

$$
P_{el} = 0.12 \cdot 7.5 = 0.9 \, \text{MW}_{el}
$$

Over a year:

$$
E = 0.9 \cdot 8760 = 7884 \, \text{MWh/year}
$$

---

# Reinforcement Learning Approach <a name="rl-main"></a>

The geothermal well control problem is modeled as a **Markov Decision Process (MDP)**:

- **State ($s_t$)**: wellhead pressure, flow rate, production temperature, demand profile  
- **Action ($a_t$)**: adjust injection temperature, valve opening, or pump speed  
- **Reward ($r_t$)**: penalizes mismatch between output and demand, while regularizing operational safety  
- **Policy ($\pi$)**: maps observed states to control actions  

The objective is to maximize expected long-term return:

$$
J(\pi) = \mathbb{E}_{\pi}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

where $\gamma$ is the discount factor.

---

# Python Implementation <a name="python-main"></a>

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
