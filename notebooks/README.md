# Reinforcement Learning Teaching Notebooks

This collection of Jupyter notebooks provides a comprehensive introduction to reinforcement learning algorithms, designed for educational purposes at the university level.

## Notebooks Overview

### 1. Dynamic Programming (`dynamic_programming.ipynb`)
- **Policy Evaluation**: Computing state-value functions using iterative methods
- **Policy Iteration**: Alternating between evaluation and improvement
- **Value Iteration**: Direct computation of optimal value functions
- **Key Concepts**: Bellman equations, convergence guarantees, optimal policies

### 2. Monte Carlo Methods (`monte_carlo_methods.ipynb`)
- **Monte Carlo Prediction**: Learning value functions from experience
- **Monte Carlo Control**: Learning optimal policies through exploration
- **Exploration vs Exploitation**: Epsilon-greedy strategies
- **Key Concepts**: Sample-based learning, episodic tasks, exploration strategies

### 3. Temporal Difference Learning (`temporal_difference_learning.ipynb`)
- **TD(0) Prediction**: Learning from incomplete episodes
- **SARSA**: On-policy temporal difference control
- **Q-Learning**: Off-policy temporal difference control
- **Key Concepts**: Bootstrapping, temporal differences, on/off-policy learning

### 4. Deep Q-Learning (`deep_q_learning.ipynb`)
- **Function Approximation**: Using neural networks to approximate Q-functions
- **Experience Replay**: Storing and sampling from past experiences
- **Target Networks**: Using separate networks for stable learning targets
- **Key Concepts**: Deep Q-Network (DQN), convergence challenges, high-dimensional state spaces

### 5. Taxi Tutorial (`openai_gym_taxi_tutorial.ipynb`)
- **Environment Setup**: Introduction to Gymnasium environments
- **Q-Learning Implementation**: Complete working example
- **Visualization**: Understanding agent behavior
- **Key Concepts**: Practical implementation, environment interaction, reward systems

## Prerequisites

- Python 3.8+
- Basic understanding of probability and statistics
- Familiarity with Python programming
- Basic knowledge of Markov Decision Processes (MDPs)

## Installation

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Environment Requirements

- **FrozenLake**: Used in most notebooks for grid-world examples
- **Taxi**: Used in the tutorial for a more complex environment
- All environments use **Gymnasium** (updated from deprecated OpenAI Gym)

## Learning Path

1. Start with **Dynamic Programming** to understand the theoretical foundations
2. Move to **Monte Carlo Methods** to see sample-based learning
3. Study **Temporal Difference Learning** for more efficient online learning
4. Explore **Deep Q-Learning** for high-dimensional state spaces
5. Complete the **Taxi Tutorial** for hands-on implementation experience

## Key Features

- **Updated Dependencies**: All notebooks use modern Gymnasium instead of deprecated Gym
- **Comprehensive Documentation**: Detailed explanations of algorithms and concepts
- **Visualizations**: Clear plots and diagrams to aid understanding
- **Interactive Examples**: Hands-on code that students can modify and experiment with
- **Progressive Complexity**: From simple grid worlds to more complex environments

## Educational Use

These notebooks are designed for:
- University-level reinforcement learning courses
- Self-study and independent learning
- Research and experimentation
- Algorithm comparison and analysis

## Contributing

Feel free to submit issues or pull requests to improve these teaching materials. Suggestions for additional examples, clearer explanations, or new algorithms are welcome.

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- OpenAI Gym Documentation: https://gymnasium.farama.org/
- Reinforcement Learning Course Materials from various universities
