"""Inference from the student's actual checkpoint; no training required."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate import load_saved, occlude, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--example', choices=['success','failure'])
    parser.add_argument('--index', type=int, default=0, help='Position in saved test split, 0..359')
    parser.add_argument('--mask-size', type=int, choices=[0,1,2,3], default=0)
    parser.add_argument('--output', default='demo.png')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, cp, images, labels, splits = load_saved(args.results)
    test = splits['test']
    pos, size = args.index, args.mask_size
    if args.example:
        guesses = predict(model, images[test]).argmax(1)
        selected = np.flatnonzero((guesses == labels[test]) if args.example == 'success' else (guesses != labels[test]))
        if len(selected) == 0:
            raise SystemExit('No example of the requested type exists; do not invent one.')
        pos, size = int(selected[0]), 0
    if not 0 <= pos < len(test):
        raise SystemExit('index must be between 0 and 359')
    idx = int(test[pos])
    original = images[idx:idx+1]
    masked = occlude(original, size)
    scores = predict(model, masked)[0]
    guess = int(scores.argmax())
    fig, axes = plt.subplots(1,3,figsize=(11,3))
    for ax,img,title in zip(axes[:2],(original[0],masked[0]),
                            (f'Original: label {labels[idx]}',f'Mask {size}: prediction {guess}')):
        ax.imshow(img,cmap='gray',vmin=0,vmax=1)
        ax.set_title(title)
        ax.axis('off')
    axes[2].bar(range(10),scores)
    axes[2].set(xticks=range(10),ylim=(0,1),title='Softmax scores (uncalibrated)')
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    plt.close(fig)
    print(f'Dataset index={idx}, test position={pos}, label={labels[idx]}, prediction={guess}, mask={size}')
    print('Image saved:',args.output)


if __name__ == '__main__':
    main()
