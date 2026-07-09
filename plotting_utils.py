import numpy as np
import matplotlib.pyplot as plt


def plot_value_comparison(states, series_dict, title="Value comparison"):
    plt.figure(figsize=(8, 5))
    for name, values in series_dict.items():
        plt.plot(states, values, marker="o", label=name)
    plt.xlabel("State")
    plt.ylabel("Estimated value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_history_snapshots(history, states, every=100, title="Learning progress"):
    plt.figure(figsize=(8, 5))
    for i, values in enumerate(history):
        if i % every == 0 or i == len(history) - 1:
            plt.plot(states, values, alpha=0.7, label=f"episode {i+1}")
    plt.xlabel("State")
    plt.ylabel("Estimated value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
