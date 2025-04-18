import argparse
from src.trainer.train import train
from src.trainer.eval import evaluate
from src.trainer.inference import inference
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="MODIS to Landsat Super-Resolution")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "inference"],
        required=True,
        help="Mode to run the script in: train | eval | inference",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to the configuration file (default: experiments/config.yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Execute the selected mode
    if args.mode == "train":
        train(config)
    elif args.mode == "eval":
        evaluate(config)
    elif args.mode == "inference":
        inference(config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
