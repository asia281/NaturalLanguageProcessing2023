# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

from run import evaluate_places
import argparse

def london_preds(path: str):
    preds = []
    for _ in open(args.eval_path, 'r'):
        preds.append('London')
    total, correct = evaluate_places(args.eval_copus_path, preds)

    print('London baseline accuracy: %.2f' % (correct / total))

argp = argparse.ArgumentParser()
argp.add_argument('--eval_path',
    help="Path of the data to evaluate model on", default=None)
args = argp.parse_args()
london_preds(args.eval_path)