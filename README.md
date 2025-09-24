# PIRO: Proximal Inverse Reward Optimization


This repository contains the implementation of PIRO (Proximal Inverse Reward Optimization), an advanced stable inverse reinforcement learning algorithm. 

## 🛠 Installation

### Requirements
- Python 3.7+
- PyTorch 1.5+
- OpenAI Gym
- [MuJoCo]
- Gymnasium Robotics

### Setup
```bash
pip install -r requirements.txt
pip install ruamel.yaml
```

## 📁 File Structure

- **PIRO implementation**: `train/`
  - `trainPIRO.py` - Main PIRO training script
  - `trainML.py` - One of baselines -- ML-IRL training 
  - `models/` - Reward function models
- **SAC agent**: `common/`
- **Environments**: `envs/`
- **Configurations**: `configs/`
- **Utilities**: `utils/`
- **Baseline methods**: `baselines/`

## 🚀 Usage

### Environment Setup
Before running experiments, set the Python path:
```bash
export PYTHONPATH=${PWD}:$PYTHONPATH
```

### Expert Data
During the review process, expert demonstrations from Hugging Face and the D4RL benchmark can be used to reproduce our results. The expert trajectories collected in this study will be released after review process concludes.

Alternatively, generate your own expert data:
```bash
# Train expert policy
python common/train_gd.py configs/samples/experts/{env}.yml

# Collect expert demonstrations  
python common/collect.py configs/samples/experts/{env}.yml

#Collect minari dataset
python common/collect_robotic.py
```
where `{env}` is one of: `hopper`, `walker2d`, `halfcheetah`, `ant`......

### Training PIRO

Train PIRO on MuJoCo and Robotic environments:
```bash
python train/trainPIRO.py configs/samples/agents/{env}.yml
```

## 📊 Results

Training logs and models are saved in the `logs/` directory with the following structure:
```
logs/{environment}/exp-{expert_episodes}/{method}/{timestamp}/
├── progress.csv          # Training metrics
├── model/               # Saved reward models  
├── variant.json         # Configuration used
└── plt/                # Plots and visualizations
```

## 🎯 Configuration

Experiments are configured using YAML files in `configs/`. Key parameters:

- **Environment settings**: Environment name, state indices, episode length
- **Algorithm settings**: Learning rates, network architectures, training iterations
- **Evaluation settings**: Number of evaluation episodes, metrics to track


## 🔬 Baseline Comparisons

The `baselines/` directory contains implementations of several IRL methods for comparison:
- GAIL, AIRL, BC, IQ-Learn, and others
- Each baseline includes its own configuration and training scripts



Our implementation draws inspiration from the structural design of the ML-IRL framework proposed by Zeng et al. [2023], but includes significant modifications tailored to our method and experiments.



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
