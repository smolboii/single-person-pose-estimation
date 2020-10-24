from torchvision import models, transforms as T
from model import PoseEstimationModel
from tqdm import tqdm
from datetime import datetime
import argparse
import numpy as np
import torch
import random
import util
import csv
import sys
import cv2

train_path = 'res/train/'
val_path = 'res/val/'
test_path = 'res/test/'

# img_id = '1000_1234574.jpg'
# annotation = annotations[img_id]
# img = util.load_img(train_path + 'img/', img_id)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--validate", action="store_true",
                        help="tests the model against the validation data")
    parser.add_argument("-t", "--train", action="store_true",
                         help="trains the model using the training data (default mode)")
    parser.add_argument("-vis", "--visualise", action="store_true",
                        help="visualise the model's predictions against the ground truth labels")
    parser.add_argument("--weights_path", type=str,
                        help="path to pre-trained weights to load into model")
    return parser.parse_args()

def generate_batch(ids, i, img_dir_path, batch_size=32):
    id_batch = ids[i*batch_size:(i+1)*batch_size]
    img_batch = [util.load_img(img_dir_path, img_id) for img_id in id_batch]
    inputs_batch = torch.stack([util.img_to_input(img, input_dims) for img in img_batch], dim=0).to(device)
    annotations_batch = [annotations[img_id] for img_id in id_batch]

    # generate target heatmaps
    gt_heatmaps_batch = []
    gt_coords_list_batch = []
    for i_, annotation in enumerate(annotations_batch):
        gt_coords_list = util.parse_annotation(annotation)
        gt_coords_list_batch.append(gt_coords_list)
        gt_heatmaps_batch.append(util.generate_gt_heatmaps(gt_coords_list, img_batch[i_].shape, heatmap_dims).to(device))
    gt_heatmaps_batch = torch.stack(gt_heatmaps_batch, dim=0).to(device)

    return img_batch, inputs_batch, gt_heatmaps_batch, gt_coords_list_batch

args = parse_args()

if torch.cuda.is_available():
    print('Using CUDA')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

model = PoseEstimationModel().to(device)

input_dims = (256,256)
heatmap_dims = (64, 64)

if args.weights_path:
    model.load_state_dict(torch.load(args.weights_path))
elif args.visualise or args.validate:
    raise Exception('Visualise / validate modes require pre-trained weights (use --weights_path option).')

if args.validate:

    annotations = util.read_annotations(val_path + 'annotations.csv')
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    ids = list(annotations.keys())
    batch_size = 32

    running_loss = 0.0
    with torch.no_grad():
        for i in tqdm(range(0, len(ids) // batch_size), position=0, leave=True, file=sys.stdout):

            img_batch, inputs_batch, gt_heatmaps_batch, _ = generate_batch(ids, i, val_path + 'img/', batch_size)

            outputs = model(inputs_batch)
            loss = loss_fn(outputs, gt_heatmaps_batch)

            running_loss += loss.item()

    print(f'Validation mean loss (using MSELoss): {round(running_loss / len(ids), 5)}')

elif args.visualise:

    model.eval()

    annotations = util.read_annotations(val_path + 'annotations.csv')

    batch_size = 16
    img_batch, inputs_batch, _, gt_coords_batch = generate_batch(list(annotations.keys()), 0, val_path + 'img/', batch_size)

    outputs_batch = model(inputs_batch)
    guess_coords_batch = [util.output_to_joints(output, img_batch[i].shape) for i, output in enumerate(outputs_batch)]

    for i, img in enumerate(img_batch):
        util.draw_joints(img, gt_coords_batch[i], (255,0,0))
        util.draw_joints(img, guess_coords_batch[i], (0,0,255))
        cv2.imshow('test', img)
        cv2.waitKey(0)

else:  # train

    annotations = util.read_annotations(train_path + 'annotations.csv')
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    shuffled_ids = list(annotations.keys())

    batch_size = 32
    n_epochs = 150

    chkpt_interval = 225
    for epoch in range(1, n_epochs+1):
        random.shuffle(shuffled_ids)
        running_loss = 0.0
        for i in tqdm(range(0, len(shuffled_ids) // batch_size), position=0, leave=True, file=sys.stdout):

            img_batch, inputs_batch, gt_heatmaps_batch, _ = generate_batch(shuffled_ids, i, train_path + 'img/', batch_size)

            optimizer.zero_grad()

            outputs = model(inputs_batch)
            loss = loss_fn(outputs, gt_heatmaps_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i > 0 and i % chkpt_interval == 0:
                # print training info and save checkpoint model
                tqdm.write(f'Epoch #{epoch}, Batch #{i} -  Mean loss: {round(running_loss / chkpt_interval, 6)}, Time: {datetime.now()}')
                file_str = f'e{epoch+1}_b{i}_ml={round(running_loss / chkpt_interval, 6)}'
                torch.save(model.state_dict(), 'checkpoints/' + file_str + '.pt')
                running_loss = 0.0


