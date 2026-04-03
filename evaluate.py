import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.experiment import (
    BASE_DIR,
    attach_retrieval_pool,
    bootstrap_experiment,
    build_model,
    compute_metrics,
    create_dataloaders,
    get_device,
    load_taxwar_fsi_dataframe,
    predict_split,
    prepare_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a reproducible split.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = bootstrap_experiment()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint["model_name"]
    prepared = prepare_data(config, model_name)
    device = get_device()

    model = build_model(model_name, config, input_size=len(prepared.feature_columns)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    attach_retrieval_pool(model, prepared.train_feature_pool, device)

    loaders = create_dataloaders(prepared, batch_size=int(config["training"]["batch_size"]))
    split_output = predict_split(model, loaders[args.split], device, prepared.scaler_y)
    metrics = compute_metrics(split_output["y_true"], split_output["y_pred"])

    criterion = nn.MSELoss()
    loss_terms = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loaders[args.split]:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x).view(-1, 1)
            loss_terms.append(float(criterion(output, batch_y).cpu().item()))
    metrics["loss"] = sum(loss_terms) / len(loss_terms) if loss_terms else float("nan")

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "split": args.split,
        "metrics": metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
