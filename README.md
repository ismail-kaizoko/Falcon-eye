# Falcon-EYE 🧿

Falcon-EYE is a personal project focused on implementing core algorithms for
vision-based navigation, tracking, and simulation in UAV (drone) systems.


---

## Implemented Topics

- **optical flow** : Lukas_Kanade algorithm to estimate local motion 
- **Epipolar geometry and Essential matrix** : estimate (R,t) Rotation translation motion between two frames.
---

##  Simulation Environment

All algorithms are tested inside **AirSim**, a high-fidelity drone simulator from Microsoft.



### 1. Install Unreal Engine

Download Unreal Engine (recommended 4.27):
https://www.unrealengine.com/


### 2. Run AirSim (fastest method)

Instead of building AirSim, you can use a prebuilt environment:

1. Visit: https://github.com/microsoft/AirSim/releases
2. Get: `Blocks.zip`
3. Extract and run:

This launches a ready-to-use 3D simulation with a drone.


clone The AirsSim repository from microsoft repo : 
```bash
git clone https://github.com/microsoft/AirSim.git
cd AirSim/PythonClient/multirotor 
python hello_drone.py
```


## Clone and Setup
AirSim dependencies may conflict with your system packages, we recommend using python env : 

```bash
git clone <repo>
cd Falcon-eye

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```