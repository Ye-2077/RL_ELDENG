# RL ELDEN RING
This project is a reinforcement learning plateform for the boss battle of game Elden Ring, based on Gymnasium and Baselines3, and the algorithm used is PPO. The game exe is not included in this project.

This project used scripts to simulate the keyboard inputs, so it can be transformed to any game that can be played with a keyboard actually.

Since this project based on the original game, ervery time the character dies, the game have to be restarted through the script. So, for different bosses, the script have to be modified to restart the game. The scripts in this project is for the boss battle of "knight of the Crucible" at "Stormhill".

**Environment and Tools**
- Python 3.10.10
- PyTorch 2.2.1+cu118
- Windows 10
- Elden Ring

## Installation
1. clone the project
2. create a virtual environment (adviced)
```
conda create -n eldenring python=3.10.10
```
3. activate the virtual environment
```
conda activate eldenring
```
4. install the requirements
```
pip install -r requirements.txt
```

## Prepare the game
**Test Vision**
1. Set the game to windowed mode
2. Using resolution ratio 1920x1080
3. Run file `test_vision.py`, and check the vision of the game is correct.

**Martch the Restart Scripts**
1. Go to the boss battle, in this project, it is "knight of the Crucible" at "Stormhill"
2. Activate the boss battle, die one time, choose "continue at Stakes of Marika".
3. Press "ESC" to pause the game, make sure the game perspective is the original perspective after death.

## Run the project
Run file `train.py` to start the training process.
