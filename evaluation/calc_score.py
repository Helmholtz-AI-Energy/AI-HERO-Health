import numpy as np
import pandas as pd
import os
import json
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str,
                        default='/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/submission_test/predictions.csv')
    parser.add_argument('--gt', type=str, default='/hkfs/work/workspace/scratch/im9193-health_challenge/data/valid.csv')
    parser.add_argument('--save_dir', type=str, default='/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/submission_test')
    args = parser.parse_args()

    print('Calculating Accuracy')

    pred_path = args.preds
    gt_path = args.gt
    save_dir = args.save_dir

    preds_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    assert len(preds_df) == len(gt_df), 'Dataframes do not have same length!'

    merged = pd.merge(preds_df, gt_df, on='image')
    merged["gt"] = np.where(merged['label'] == 'positive', 1, 0)

    acc = 100. * (sum(merged["prediction"] == merged["gt"]) / len(merged))

    print('Done! Accuracy:', acc)

    result_path = os.path.join(save_dir, 'score.json')
    with open(result_path, 'w') as f:
        json.dump(acc, f)

    print('The result is saved in {}'.format(result_path))
