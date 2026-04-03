import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.experiment import (
    attach_retrieval_pool,
    bootstrap_experiment,
    build_model,
    compute_metrics,
    create_dataloaders,
    get_device,
    prepare_data,
    predict_split,
    run_epoch,
    save_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with fixed reproducibility settings.")
    parser.add_argument("--model", required=True, choices=["ragformer", "lstm", "dtml", "transformer", "master"])
    parser.add_argument("--run-name", default=None, help="Artifact basename. Defaults to <model>_repro_<seed>.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = bootstrap_experiment()
    prepared = prepare_data(config, args.model)
    device = get_device()

    training_cfg = config["training"]
    model = build_model(args.model, config, input_size=len(prepared.feature_columns)).to(device)
    attach_retrieval_pool(model, prepared.train_feature_pool, device)

    loaders = create_dataloaders(prepared, batch_size=int(training_cfg["batch_size"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = int(training_cfg["patience"])
    stale_epochs = 0
    grad_clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0))
    history = []

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, criterion, device, train=True, grad_clip_norm=grad_clip_norm)
        val_metrics = run_epoch(model, loaders["validation"], optimizer, criterion, device, train=False)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch} after {patience} stale epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    attach_retrieval_pool(model, prepared.train_feature_pool, device)

    final_metrics = {}
    for split_name, loader in loaders.items():
        scaled_metrics = run_epoch(model, loader, optimizer, criterion, device, train=False)
        split_output = predict_split(model, loader, device, prepared.scaler_y)
        raw_metrics = compute_metrics(split_output["y_true"], split_output["y_pred"])
        final_metrics[split_name] = {
            "scaled": scaled_metrics,
            "raw": raw_metrics,
        }

    final_metrics["training_summary"] = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "device": str(device),
    }

    run_name = args.run_name or f"{args.model}_repro_{config['seed']['global_seed']}"
    artifact_paths = save_artifacts(
        model=model,
        model_name=args.model,
        prepared=prepared,
        config=config,
        run_name=run_name,
        metrics=final_metrics,
        feature_columns=prepared.feature_columns,
    )

    print(json.dumps({"run_name": run_name, "artifacts": artifact_paths, "metrics": final_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
