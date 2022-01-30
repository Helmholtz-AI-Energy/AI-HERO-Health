import os
import requests
import json
from argparse import ArgumentParser
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from dataset import CovidImageDataset

# TODO: import your model
from model import VGG as SubmittedModel


def predict(model, device, test_loader):
    model.eval()
    model.to(device)
    predictions = []
    with torch.no_grad():
        for data, img_name in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            predictions.append([[i.split('/')[-1] for i in img_name], predicted.cpu().numpy()])

    return predictions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str, default='/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/saved_models/vgg_baseline.pt',
                        help="Model weights path")  # TODO: adapt to your model weights path in the bash script
    parser.add_argument("--save_dir", type=str, help='Directory where weights and results are saved',
                        default='/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/submission_test')
    parser.add_argument("--data_dir", type=str, help='Directory containing the data you want to predict',
                        default='/hkfs/work/workspace/scratch/im9193-health_challenge')
    args = parser.parse_args()

    weights_path = args.weights_path
    save_dir = args.save_dir
    data_dir = args.data_dir

    os.makedirs(save_dir, exist_ok=True)

    check_script = 'test_data' not in data_dir

    filename = weights_path.split('/')[-1]

    # load model with pretrained weights
    model = SubmittedModel('VGG19')  # TODO: adjust arguments according to your model
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # dataloader
    data_split = 'validation' if check_script else 'test'
    print('Running inference on {} data'.format(data_split))
    testset = CovidImageDataset(
        os.path.join(data_dir, 'evaluation/{}.csv'.format('valid' if check_script else 'test')),
        os.path.join(data_dir, 'data/imgs' if check_script else 'data/test'),
        transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=128)

    # run inference
    device = torch.device("cuda")

    predictions = predict(model, device, testloader)
    flattened_preds = [np.concatenate(i) for i in list(zip(*predictions))]
    pd.DataFrame(data={'image': flattened_preds[0], 'prediction': flattened_preds[1]}).to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

    print('Done! The result is saved in {}'.format(save_dir))

