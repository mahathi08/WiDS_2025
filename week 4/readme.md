# Teaching AI Agents to Walk — REINFORCE on Humanoid  
WiDS 5.0 Final Project — Reinforcement Learning

This project implements a **REINFORCE (Monte-Carlo Policy Gradient)** algorithm from scratch using **PyTorch** and **Gymnasium** to train a humanoid robot in a MuJoCo environment to walk forward while maintaining balance.

The agent uses a neural network Gaussian policy and is trained using episodic returns.

It explains visually : 
- Model-free policy gradient learning
- Continuous action control
- Neural network policy parameterization
- MuJoCo humanoid simulation
- Gymnasium environment usage


# (Steps to follow::::::::::::::::::::::::)

python -m venv venv
venv\Scripts\activate        # Windows
# OR
source venv/bin/activate     # Mac/Linux

pip install --upgrade pip
pip install gymnasium[mujoco] torch numpy matplotlib

python training.py
What happens:
        Loads Humanoid-v4 environment
        Builds neural policy
        Runs REINFORCE training
        Prints rewards every 10 episodes
        Saves trained weights → **model**
        Displays reward curve plot
        Training is compute heavy — may take time.

python check.py
        This will:
        Load saved model
        Open MuJoCo humanoid renderer
        Run learned policy
        Show walking behavior
        Rendering uses deterministic mean actions for stability.


note::::.............................
    While running training.py with episodes=500 ,it is largely time taking....To just check this part you may reduce it to lower number.After it gives the file ""model"" you run check.py to visualize the movement of humanoid.
