"""Training exported from capstone_digits.ipynb; same split, model and hyperparameters.
No test evaluation or hyperparameter search is performed here.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--out', default='results_reproduced')
    args = parser.parse_args()
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import json
    import random
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output is not empty. Use a new --out; saved student results must not be overwritten.")
    out.mkdir(parents=True, exist_ok=True)

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    digits = load_digits()

    # Normalize ด้วยขอบเขตค่าที่ทราบจากคำอธิบายข้อมูล
    # ไม่ได้คำนวณค่าสถิติจาก test
    X = (digits.images / 16.0).astype(np.float32)
    y = digits.target.astype(np.int64)

    indices = np.arange(len(y))

    train_idx, remaining_idx = train_test_split(
        indices,
        test_size=0.4,
        random_state=seed,
        stratify=y
    )

    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=0.5,
        random_state=seed,
        stratify=y[remaining_idx]
    )

    assert not set(train_idx) & set(val_idx)
    assert not set(train_idx) & set(test_idx)
    assert not set(val_idx) & set(test_idx)

    np.savez(
        out / "split_indices.npz",
        train=train_idx,
        val=val_idx,
        test=test_idx
    )

    for name, idx in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx)
    ]:
        print(name, len(idx))
        print("class counts:", np.bincount(y[idx], minlength=10))

    fig, axes = plt.subplots(2, 5, figsize=(9, 4))

    for ax, idx in zip(axes.ravel(), train_idx[:10]):
        ax.imshow(X[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Label: {y[idx]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out / "train_examples.png", dpi=160)
    plt.close()

    from model import SmallCNN

    model = SmallCNN(cfg["dropout"]).to(device)

    print(model)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    import copy
    import csv

    train_dataset = TensorDataset(
        torch.from_numpy(X[train_idx, None]),
        torch.from_numpy(y[train_idx])
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    val_x = torch.from_numpy(X[val_idx, None]).to(device)
    val_y = torch.from_numpy(y[val_idx]).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        loss_sum = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(batch_y)
            correct += (
                logits.argmax(dim=1) == batch_y
            ).sum().item()
            total += len(batch_y)

        model.eval()

        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()
            val_accuracy = (
                val_logits.argmax(dim=1) == val_y
            ).float().mean().item()

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "val_loss": val_loss,
            "train_accuracy": correct / total,
            "val_accuracy": val_accuracy
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={row['train_loss']:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val accuracy={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= cfg["patience"]:
            print("Early stopping")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    # เก็บน้ำหนักไว้บน CPU เพื่อโหลดสะดวกทั้ง CPU/GPU
    cpu_state = {
        key: value.detach().cpu()
        for key, value in best_state.items()
    }

    torch.save(
        {
            "state_dict": cpu_state,
            "config": cfg,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss
        },
        out / "best_model.pt"
    )

    with open(out / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print("Selected epoch:", best_epoch)

    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(
        epochs, [r["train_loss"] for r in history],
        label="Train"
    )
    axes[0].plot(
        epochs, [r["val_loss"] for r in history],
        label="Validation"
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(
        epochs, [r["train_accuracy"] for r in history],
        label="Train"
    )
    axes[1].plot(
        epochs, [r["val_accuracy"] for r in history],
        label="Validation"
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out / "learning_curves.png", dpi=160)
    plt.close()

    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    import joblib

    flat_train = X[train_idx].reshape(len(train_idx), -1)

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(flat_train, y[train_idx])

    logistic = LogisticRegression(
        C=1.0,
        max_iter=3000,
        random_state=seed
    )
    logistic.fit(flat_train, y[train_idx])

    joblib.dump(
        {"majority": majority, "logistic": logistic},
        out / "baselines.joblib"
    )

    import platform
    import sklearn
    import matplotlib
    import joblib
    (out / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "joblib": joblib.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None
    }, indent=2))
    print("Training complete. Test predictions have NOT been evaluated. Output:", out)


if __name__ == '__main__':
    main()
