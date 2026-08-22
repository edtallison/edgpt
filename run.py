import torch

from config import GPTConfig
from gpt import GPT


def run():
    config = GPTConfig()
    model = GPT(config)

    x = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(x)
    print(logits.shape)  # torch.Size([2, 10, 50257])
    print(loss)  # None, no targets given

    targets = torch.randint(0, config.vocab_size, (2, 10))
    logits, loss = model(x, targets)
    print(loss)  # a scalar tensor, should be roughly log(vocab_size) ≈ 10.8 for an untrained model


if __name__ == "__main__":
    run()
