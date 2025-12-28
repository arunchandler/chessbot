# chessbot

Making a chess bot with python-chess, MCTS, and a neural network.

## Commands and options

- Self-play data: `python scripts/selfplay.py --games N --output data.pt [--model-path MODEL --device cpu|cuda --iterations 400 --cpuct 1.5 --temperature 1.0 --temperature-final 0.0 --temperature-moves 20 --dirichlet-alpha 0.3 --dirichlet-frac 0.25 --buffer-path replay.pkl --buffer-capacity 200000]`
- Train: `python scripts/train.py --data data.pt --epochs N --checkpoint model1.pt [--batch-size 64 --lr 1e-3 --weight-decay 1e-4 --device cpu|cuda --resume CKPT]`
- Train from buffer: `python scripts/train.py --buffer replay.pkl [--buffer-samples 0 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --device cpu|cuda --epochs N --checkpoint model1.pt --resume CKPT]`
- Auto loop (self-play → buffer → train → gated eval): `python scripts/auto_loop.py [--iterations 400 --games 10 --buffer-path replay.pkl --buffer-capacity 200000 --buffer-samples 50000 --epochs 1 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --train-device cpu|cuda --model-device cpu|cuda --best-checkpoint checkpoints/best.pt --candidate-checkpoint checkpoints/candidate.pt --eval-games 20 --gating-threshold 0.55 --iterations-count 1]`
- Play vs bot: `python scripts/play.py --bot random|mcts [--model-path MODEL --model-device cpu|cuda --iterations 300 --cpuct 1.5 --color white|black]`
- UCI engine: `python scripts/uci_engine.py [--model-path MODEL --device cpu|cuda --iterations 400 --cpuct 1.5]`
- Eval vs random: `python scripts/eval_random.py --games N [--model-path MODEL --device cpu|cuda --iterations 300 --cpuct 1.5]`
- Eval vs material MCTS: `python scripts/eval_material.py --games N [--model-path MODEL --device cpu|cuda --iterations 300 --cpuct 1.5 --baseline-iterations 300]`
- Run tests: `pytest tests`

Note: Training data (`data.pt`, etc.) and model checkpoints (`model1.pt`, etc.) can be large; keep them out of git and store them in external storage (e.g., local disk outside the repo, cloud bucket, or git-lfs).
