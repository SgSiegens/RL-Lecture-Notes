import torch
from transformers import GPT2Model, GPT2Tokenizer
import seaborn as sns
import matplotlib.pyplot as plt


def format_vertical_math_label(token, i):
    return f"{token}\n" \
           f"$\\downarrow$\n" \
           f"$\\vec{{E}}_{{{i}}}$\n" \
           f"$\\downarrow$\n" \
           f"$\\vec{{Q}}_{{{i}}}$"


def make_attention_plot(attn_pattern, tokens, path):
    fig = plt.figure(figsize=(15, 10))
    key_tokens = [f"{tok}$\\rightarrow \\vec{{E}}_{{{i}}} \\rightarrow \\vec{{K}}_{{{i}}}$" for i, tok in enumerate(tokens)]
    query_tokens = [format_vertical_math_label(tok, i) for i, tok in enumerate(tokens)]
    ax = sns.heatmap(attn_pattern, xticklabels=query_tokens, yticklabels=key_tokens, cmap="viridis")
    ax.set_aspect('auto')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
    #fig.savefig(path, bbox_inches='tight')
    #print(f"Saved attention plot to {path}")


def main():
    model_name = "gpt2"
    sentence = "Kings made laws to keep power over the small towns."

    # Load model and tokenizer
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()

    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Get embeddings and position ids
    input_embeds = model.wte(input_ids)
    position_ids = torch.arange(0, input_ids.size(-1)).unsqueeze(0)
    position_embeds = model.wpe(position_ids)

    # Add embeddings and layer norm
    hidden_states = input_embeds + position_embeds
    hidden_states = model.drop(hidden_states)
    hidden_states = model.h[0].ln_1(hidden_states)

    # Extract raw Q, K, V from first attention block
    attn = model.h[0].attn
    qkv = attn.c_attn(hidden_states)
    batch_size, seq_len, _ = qkv.shape
    qkv = qkv.view(batch_size, seq_len, attn.num_heads, 3 * attn.head_dim)
    qkv = qkv.permute(0, 2, 1, 3)
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    # Compute raw attention scores (QK^T)
    attn_scores = torch.matmul(q, k.transpose(-1, -2))[0, 0]

    # Apply softmax with causal masking
    d_k = q.shape[-1]
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=bool), diagonal=1)
    mod_attn_scores = attn_scores / d_k ** 0.5
    mod_attn_scores[mask] = float('-inf')
    softmax_attn = torch.softmax(mod_attn_scores, dim=1).detach().numpy()

    # Token labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [t.lstrip('Ġ') for t in tokens]  # Remove GPT-2 space marker

    # Plot both raw and modified attention
    make_attention_plot(attn_scores.detach().numpy(), tokens, "normal_attn.pdf")
    make_attention_plot(softmax_attn, tokens, "mod_attn.pdf")


if __name__ == "__main__":
    main()
