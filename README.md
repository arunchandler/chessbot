# chessbot

Making a chess bot with python-chess, MCTS, and a neural network.

## Commands and options

- Self-play data: `python scripts/selfplay.py --games N --output data.pt [--model-path MODEL --device cpu|cuda --iterations 400 --cpuct 1.5 --temperature 1.0 --temperature-moves 20]`
- Train: `python scripts/train.py --data data.pt --epochs N --checkpoint model1.pt [--batch-size 64 --lr 1e-3 --weight-decay 1e-4 --device cpu|cuda --resume CKPT]`
- Play vs bot: `python scripts/play.py --bot random|mcts [--model-path MODEL --model-device cpu|cuda --iterations 300 --cpuct 1.5 --color white|black]`
- UCI engine: `python scripts/uci_engine.py [--model-path MODEL --device cpu|cuda --iterations 400 --cpuct 1.5]`
- Eval vs random: `python scripts/eval_random.py --games N [--model-path MODEL --device cpu|cuda --iterations 300 --cpuct 1.5]`
- Eval vs material MCTS: `python scripts/eval_material.py --games N [--model-path MODEL --device cpu|cuda --iterations 300 --cpuct 1.5 --baseline-iterations 300]`
