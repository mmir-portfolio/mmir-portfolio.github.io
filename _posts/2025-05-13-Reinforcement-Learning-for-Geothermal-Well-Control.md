---
layout: post
title: Reinforcement Learning for Geothermal Well Control
image: "/posts/geothermal-rl-title-img.png"
tags: [Reinforcement Learning, Energy Optimization, Geothermal Well, Python, Clean Energy]
---

- [00. Project Overview](#overview-main)  
- [01. Background](#background-main)  
- [02. Energy Harvested Calculation](#energy-main)  
- [03. Reinforcement Learning Approach](#rl-main)  
- [04. Python Implementation](#python-main)  
- [05. Results and Insights](#results-main)  

---

# Project Overview <a name="overview-main"></a>

This project explores how **Reinforcement Learning (RL)** can be applied to the control of closed-loop geothermal wells for optimizing thermal energy extraction.  
By dynamically adjusting injection and production strategies, RL enables systems to respond to **peak electricity demand** while maximizing long-term geothermal reservoir sustainability.

---

# Background <a name="background-main"></a>

Closed-loop geothermal systems circulate a working fluid through subsurface heat exchangers without directly extracting brine or reservoir fluids.  
Unlike conventional geothermal wells, these systems are less prone to scaling, corrosion, or depletion-related issues.

However, effective energy extraction depends on managing:

- **Flow rates**  
- **Temperature gradients**  
- **Reservoir pressure conditions**  
- **Surface demand variability**

Traditional controllers (PID, rule-based logic) may not adapt optimally to rapidly changing energy demands. RL offers a **data-driven, adaptive control method**.

---

# Energy Harvested Calculation <a name="energy-main"></a>

We estimate geothermal thermal power output as:

$$
Q = \dot{m} \cdot c_p \cdot (T_{prod} - T_{inj})
$$

Where:

- $\dot{m}$ : mass flow rate (kg/s)  
- $c_p$ : specific heat of water (~4180 J/kg·K)  
- $(T_{prod}, T_{inj})$ : production and injection temperatures (°C)  

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

The RL formulation involves:

- **Environment**: geothermal reservoir + surface plant dynamics  
- **Agent**: controller (e.g., PPO, DQN)  
- **State ($s_t$)**: flow rate, wellhead pressure, temperature gradient, current demand  
- **Action ($a_t$)**: adjust flow rates, injection temperature, or valve control  
- **Reward ($r_t$)**: balance between (i) maximizing energy output and (ii) meeting demand at minimum operational cost  

Mathematically, the agent maximizes expected return:

$$
J(\pi) = \mathbb{E}_{\pi}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

where $\pi$ is the policy and $\gamma$ is the discount factor.

---

# Python Implementation <a name="python-main"></a>

```python
import gym
import numpy as np
from stable_baselines3 import PPO

# Define custom geothermal environment
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
