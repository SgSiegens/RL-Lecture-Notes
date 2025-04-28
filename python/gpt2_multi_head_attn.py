import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer


def plot_attention(model_name="gpt2", sentence="Kings made laws to keep power over the small towns.",
                   save_path="multi_head_attn.pdf"):
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Get outputs with attention
    outputs = model(**inputs)
    attentions = outputs.attentions

    # Use attention from the first layer and first batch
    attention_matrix = attentions[0][0]  # shape: (num_heads, seq_len, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [token.lstrip('Ġ') for token in tokens]  # Remove GPT2-specific whitespace marker

    # Plot attention heads
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        sns.heatmap(attention_matrix[i].detach().numpy(),
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap="viridis",
                    ax=ax)
        ax.set_title(f"Head {i + 1}")
    plt.tight_layout()
    plt.show()

    #fig.savefig(save_path, bbox_inches='tight')
    #print(f"Attention plot saved to {save_path}")


if __name__ == "__main__":
    plot_attention()
