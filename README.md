# RL-Based-Swarm-Navigation
Research project on RL-based swarm navigation using ESP32-powered robots with encoders, wireless communication, and decentralized decision-making for cooperative path planning and obstacle avoidance.

# RL-Based Swarm Navigation

This repository contains the implementation and experiments of a **Reinforcement Learning (RL) based swarm navigation system**.  
The project focuses on enabling a group of autonomous ground robots to **collaboratively explore and navigate** through dynamic environments using **decentralized decision-making** and **real-time communication**.

---

## 🔹 Project Overview
- **Multi-Agent RL (MARL):** Training swarm robots with decentralized and cooperative policies.  
- **Odometry & Localization:** Encoder-based position and velocity estimation for each robot.  
- **ESP-NOW Communication:** Real-time peer-to-peer data exchange (positions, velocities, and obstacle information) between ESP32-based swarm units.  
- **Simulation + Real-World Validation:** Algorithms are first tested in simulation (Python/MATLAB) and then deployed on physical robots.  
- **Swarm Intelligence:** Collective behaviors emerge through shared environment knowledge and reinforcement learning policies.  

---

## 🔹 Hardware Setup
Each swarm unit (robot) consists of:
- **ESP32 (NodeMCU)** – main controller & communication  
- **Encoder-equipped DC motors (GA12-N20 recommended)** – precise odometry  
- **Motor Driver (L298N / TB6612FNG)** – motor control  
- **Ultrasonic Sensor (HC-SR04)** – obstacle detection  
- **LiPo / 18650 Battery + Protection Circuit** – power supply  
- **Acrylic/Aluminum Robot Chassis** – lightweight frame  

---

## 🔹 Software Stack
- **Arduino (C++)** – low-level control, motor drivers, sensor reading, ESP-NOW communication  
- **Python (PyTorch / TensorFlow)** – RL algorithms and training (Q-Learning, DQN, or PPO)  
- **MATLAB / Python** – simulation environment & data analysis  
- **Gazebo / Webots (Optional)** – 3D multi-robot simulation  

---

## 🔹 Communication (ESP-NOW)
- Peer-to-peer communication between up to 4 robots  
- Each robot broadcasts:
  - `(x, y, θ)` position (from odometry)  
  - `v` velocity  
  - `obstacle_info` (detected via ultrasonic sensors)  
- Shared data is used by the RL policy for decentralized navigation decisions.  

---

## 🔹 Workflow
1. **Simulation:** Train MARL policies in Python/MATLAB simulation.  
2. **Deployment:** Upload motor control + communication code to ESP32s.  
3. **Integration:** Combine RL-trained navigation policies with real swarm robots.  
4. **Evaluation:** Compare performance in simulation vs real-world experiments.  

---

## 🔹 Project Goals
- Bridge the gap between **simulation and hardware** in swarm robotics.  
- Demonstrate the feasibility of **RL for decentralized multi-robot navigation**.  
- Validate collective behaviors like **flocking, obstacle avoidance, and exploration**.  

---

## 🔹 Future Work
- Integration of **vision-based sensors (camera / LiDAR)**  
- Scaling to **more than 4 robots**  
- Advanced MARL algorithms (e.g., MADDPG, MAPPO)  
- Communication-efficient learning strategies  

---

## 🔹 Contributors
- **Ömer Yiğit** (Mechatronics Engineering, Mathematics Minor)  
- Team Members (to be added)  

---

## 🔹 License
This project is released under the MIT License.  

---
