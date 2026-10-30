"""Re-evaluate a saved run using its own held-out indices, without retraining.
Load only your own trusted baselines.joblib; joblib is a pickle-based format.
"""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from model import SmallCNN


def load_saved(folder):
    folder = Path(folder)
    checkpoint = torch.load(folder / 'best_model.pt', map_location='cpu', weights_only=True)
    model = SmallCNN(checkpoint['config']['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    data = load_digits()
    images = (data.images / 16.0).astype(np.float32)
    labels = data.target.astype(np.int64)
    with np.load(folder / 'split_indices.npz', allow_pickle=False) as saved:
        splits = {key: saved[key].copy() for key in ('train', 'val', 'test')}
    combined = np.concatenate(list(splits.values()))
    if not np.array_equal(np.sort(combined), np.arange(len(labels))):
        raise ValueError('Splits must be complete and non-overlapping')
    return model, checkpoint, images, labels, splits


def occlude(images, size):
    if size not in (0, 1, 2, 3):
        raise ValueError('mask size must be 0, 1, 2 or 3')
    result = images.copy()
    if size:
        start = (8 - size) // 2
        result[:, start:start + size, start:start + size] = 0
    return result


def predict(model, images):
    with torch.inference_mode():
        return model(torch.from_numpy(images[:, None])).softmax(1).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--output', default='evaluation_recheck.csv')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit('Output exists; choose a new --output instead of overwriting.')
    torch.set_num_threads(2)
    model, cp, images, labels, splits = load_saved(args.results)
    baselines = joblib.load(Path(args.results) / 'baselines.joblib')
    test = splits['test']
    rows = []
    for size in cp['config']['occlusion_sizes']:
        masked = occlude(images[test], size)
        flat = masked.reshape(len(test), -1)
        predictions = {
            'Majority': baselines['majority'].predict(flat),
            'LogisticRegression': baselines['logistic'].predict(flat),
            'CNN': predict(model, masked).argmax(1),
        }
        for name, guessed in predictions.items():
            rows.append(dict(model=name, occlusion_size=size,
                             accuracy=float(accuracy_score(labels[test], guessed)),
                             macro_f1=float(f1_score(labels[test], guessed, labels=list(range(10)),
                                                    average='macro', zero_division=0))))
    with open(args.output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print('Saved:', args.output, '| Original results/metrics.csv has not been modified.')


if __name__ == '__main__':
    main()
