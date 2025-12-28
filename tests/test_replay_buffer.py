import torch

from chessbot.selfplay.buffer import ReplayBuffer
from chessbot.selfplay.dataset import SelfPlayDataset, make_dataloader


def test_replay_buffer_add_and_len() -> None:
    buf = ReplayBuffer(capacity=3)
    ex = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    buf.add([ex])
    assert len(buf) == 1
    buf.add([ex, ex, ex])
    assert len(buf) == 3  # capped at capacity


def test_replay_buffer_save_load(tmp_path) -> None:
    buf = ReplayBuffer(capacity=2)
    ex = (torch.ones(1), torch.ones(1), torch.ones(1))
    buf.add([ex, ex])
    path = tmp_path / "buf.pkl"
    buf.save(path)

    loaded = ReplayBuffer.load(path)
    assert len(loaded) == 2
    a, b = loaded.sample(2)
    assert torch.equal(a[0], torch.ones(1))
    assert torch.equal(b[1], torch.ones(1))
    # sampling should be random, but all entries are identical so equality holds


def test_dataset_and_dataloader() -> None:
    examples = [
        (torch.zeros((18, 8, 8)), torch.zeros(10), torch.tensor([0.0])),
        (torch.ones((18, 8, 8)), torch.ones(10), torch.tensor([1.0])),
    ]
    ds = SelfPlayDataset(examples)
    assert len(ds) == 2
    x, pi, z = ds[0]
    assert x.shape == (18, 8, 8)
    dl = make_dataloader(examples, batch_size=2, shuffle=False)
    batch = next(iter(dl))
    assert batch[0].shape[0] == 2
