import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pos_enc(d, t):
    assert d % 2 == 0, "Dimension `d` must be even."
    encoding = np.zeros(d)
    for i in range(0, d, 2):
        freq = 1 / (10000 ** (i / d))
        encoding[i] = np.sin(t * freq)
        encoding[i + 1] = np.cos(t * freq)
    return encoding


def plot_positional_encodings(d_model=120, num_positions=50, save_path="pos_enc_visual.pdf"):
    encodings = np.array([pos_enc(d_model, t) for t in range(num_positions)])
    fig = plt.figure(figsize=(16, 6))
    sns.heatmap(data=encodings, cmap="mako", cbar=True)
    plt.title(f"Positional Encodings (dim={d_model}) Across Positions", fontsize=14)
    plt.xlabel("Encoding Dimension")
    plt.ylabel("Position (t)")
    plt.tight_layout()
    plt.xticks(ticks=np.arange(0, d_model, 20), labels=np.arange(0, d_model, 20))
    plt.yticks(ticks=np.arange(0, num_positions, 10), labels=np.arange(0, num_positions, 10))
    plt.show()
    #fig.savefig(save_path)
    #print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    plot_positional_encodings()
