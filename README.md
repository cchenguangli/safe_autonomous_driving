# CARLA SAC Reinforcement Learning

This project implements **Safe Autonomous Driving** in the CARLA simulator using 
**Soft Actor-Critic (SAC)** with a custom Gymnasium environment. 
The agent learns point-to-point navigation in urban environments with:

- Lane keeping 
- Curvature-aware speed control 
- NPC vehicles 
- Traffic lights 
- Stop signs 
- Pedestrian crossing events 

---

## 📁 Project Structure

<pre>
carla_rl/
│── sac_env.py        # Custom CARLA RL environment
│── train.py          # Training script (SAC)
│── enjoy.py          # Evaluation script with HUD + camera
│── requirements.txt  # Python dependencies
│── LICENSE           # MIT License
└── README.md
</pre>

> ⚠️ Important: The CARLA simulator files (CarlaUE4, Engine, HDMaps, etc.) 
should NOT be uploaded to GitHub.

---

## 🛠 Requirements

### 🔹 Python Version  
This project is tested on **Python 3.7**.

### 🔹 CARLA Version  
This project uses **CARLA 0.9.15**. 
Ensure your CARLA installation matches this version.

### 🔹 Python Packages  
The required Python packages are listed in:

<pre>
requirements.txt
</pre>

Install dependencies with:

<pre>
pip install -r requirements.txt
</pre>

---

## 🚗 Training

Run the SAC training script:

<pre>
python train.py
</pre>

Training logs and model checkpoints are saved to:

<pre>
./logs_sac/
</pre>

---

## 🎮 Evaluation / Enjoy Mode

Run the trained agent visually in CARLA:

<pre>
python enjoy.py --model-path logs_sac/best_model.zip
</pre>

Example:

<pre>
python enjoy.py --model-path logs_sac/best_model.zip --draw-world
</pre>

---

## 📝 Notes

Before training or evaluation, start CARLA server:

<pre>
./CarlaUE4.sh
</pre>

Environment automatically handles:

- NPC vehicle spawning 
- Pedestrian crossings 
- Traffic lights 
- Stop signs 
- Route replanning 

---

## 📄 License

This project is released under the **MIT License**.

---

## 🙌 Acknowledgements

Thanks to the CARLA Simulator team and the Stable-Baselines3 developers.

