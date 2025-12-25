import pytest

torch = pytest.importorskip("torch")

from chessbot.nn.model import ChessNet
from chessbot.nn import encode
from chessbot.nn.optim import make_optimizer
from chessbot.selfplay.dataset import make_dataloader
from chessbot.train import train as train_mod
from chessbot.train import checkpoints


def test_smoke_train_checkpoint(tmp_path) -> None:
    model = ChessNet(channels=32, num_blocks=1)
    opt = make_optimizer(model.parameters(), lr=1e-3)

    # Dummy data: two examples
    x = torch.zeros((2, encode.PLANE_COUNT, 8, 8), dtype=torch.float32)
    pi = torch.full((2, encode.ACTION_DIM), 1.0 / encode.ACTION_DIM)
    z = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    dl = make_dataloader([(x[0], pi[0], z[0]), (x[1], pi[1], z[1])], batch_size=2, shuffle=False)
    batch = next(iter(dl))

    metrics = train_mod.train_step(model, opt, batch)
    assert "loss" in metrics

    ckpt_path = tmp_path / "ckpt.pt"
    checkpoints.save_checkpoint(ckpt_path, model=model, optimizer=opt, step=1, config={"foo": "bar"})

    model2 = ChessNet(channels=32, num_blocks=1)
    opt2 = make_optimizer(model2.parameters(), lr=1e-3)
    meta = checkpoints.load_checkpoint(ckpt_path, model=model2, optimizer=opt2)
    assert meta["step"] == 1

    # Forward still works
    policy_logits, value = model2(x)
    assert policy_logits.shape == (2, encode.ACTION_DIM)
    assert value.shape == (2, 1)
