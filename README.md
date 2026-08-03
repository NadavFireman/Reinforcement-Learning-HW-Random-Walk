# Reinforcement Learning - Random Walk Prediction

**Home Assignment (Grade 100, M.Sc. Data Science, HIT). Estimating the state-value function of the classic Random Walk — a 7-state chain with known closed-form values, so every estimate is measured against exact ground truth. Five methods implemented from scratch (NumPy only): First-visit Monte Carlo, TD(0), linear approximation with one-hot and polynomial features, and a small PyTorch value network (bonus).**

## The Five Methods (RMSE vs. ground truth, 1,000 episodes)
- **First-visit Monte Carlo: 0.037** — the winner in this run; unbiased averaging pays off in a small environment.
- **Linear FA, one-hot features: 0.087.**
- **Tabular TD(0): 0.134.**
- **Linear FA, polynomial features: 0.163** — global weight coupling adds variance.
- **Neural network (PyTorch, bonus): ~0.23** — in a simple linear environment, extra capacity doesn't pay.

## Key Features
- **Theory to Simulation:** The Markov property, the Bellman equation and manually derived state values — verified in simulation.
- **Bias–Variance & Features:** Learning curves contrasting MC's stable convergence with TD's bootstrapping oscillations; one-hot vs. polynomial representation, and when function approximation becomes necessary ("the deadly triad").
- **Error Analysis:** Per-state absolute error across all methods, including terminal-state initialization effects.

## Repository Structure
- `RL_HW_RandomWalk_Notebook.ipynb`: Full solution — theory answers, all five implementations, comparison graphs and analysis.
- `random_walk_env.py`: The provided Random Walk environment (configurable chain, rewards, policies), closed-form true values and feature helpers.
- `plotting_utils.py`: Value-comparison and learning-progress plotting helpers.
- `RL_HW_RandomWalk_Instructions.md` / `Assignment_Instructions.pdf`: Assignment instructions (Markdown and PDF).
