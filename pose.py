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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--validate", action="store_true",
                        help="tests the model against the validation data")
    parser.add_argument("-t", "--train", action="store_true",
                         help="trains the model using the training data (default mode)")
    parser.add_argument("-vis", "--visualise", action="store_true",
                        help="visualise the model's predictions against the ground truth labels")
    parser.add_argument("-e", "--epoch-count", type=int,
                        help="number of epochs to train for")
    parser.add_argument("-eo", "--epoch-offset", type=int,
                        help="epoch # to start counting from (simply for convenience of file naming)")
    parser.add_argument("--weights-path", type=str,
                        help="path to pre-trained weights to load into model")
    return parser.parse_args()

def generate_batch(annotations, ids, i, img_dir_path, batch_size=32):
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

def train(n_epochs, chkpt_interval=225, val_interval=2, epoch_offset=1):

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    ids = list(train_annotations.keys())

    for epoch in range(epoch_offset, n_epochs+1):
        random.shuffle(ids)
        running_loss = 0.0
        for i in tqdm(range(0, len(ids) // batch_size), position=0, leave=True, file=sys.stdout):

            img_batch, inputs_batch, gt_heatmaps_batch, _ = generate_batch(train_annotations, ids, i, train_path + 'img/', batch_size)

            optimizer.zero_grad()

            outputs = model(inputs_batch)
            loss = loss_fn(outputs, gt_heatmaps_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i > 0 and i % chkpt_interval == 0:
                # print training info and save checkpoint model
                tqdm.write(f'Epoch #{epoch}, Batch #{i} -  Mean loss: {round(running_loss / chkpt_interval, 6)}, Time: {datetime.now()}')
                file_str = f'e{epoch}_b{i}_ml={round(running_loss / chkpt_interval, 6)}'
                torch.save(model.state_dict(), 'checkpoints/' + file_str + '.pt')
                running_loss = 0.0

        if (epoch-epoch_offset) > 0 and (epoch-epoch_offset) % val_interval == 0:
            # validate model
            validate()

def validate(epoch_size_factor = 0.3):

    # epoch_size_factor specifies what percentage of the epoch the model should be validated on(e.g. 0.3 means 30% of the
    # validation dataset)

    model.eval()

    loss_fn = torch.nn.MSELoss()

    ids = list(val_annotations)

    val_running_loss = 0.0
    n_batches = int((len(val_annotations.keys()) * epoch_size_factor) // batch_size)
    with torch.no_grad():
        for i in tqdm(range(0, n_batches), position=0, leave=True, file=sys.stdout):

            img_batch, inputs_batch, gt_heatmaps_batch, _ = generate_batch(val_annotations, ids, i, val_path + 'img/', batch_size)

            outputs = model(inputs_batch)
            loss = loss_fn(outputs, gt_heatmaps_batch)

            val_running_loss += loss.item()

    print(f'Validation mean loss (using MSELoss): {round(val_running_loss / n_batches, 5)}')

def visualise():

    model.eval()

    batch_size = 16
    rand_i = np.random.randint(len(val_annotations) // batch_size) - 1
    img_batch, inputs_batch, _, gt_coords_batch = generate_batch(val_annotations, list(val_annotations.keys()), rand_i, val_path + 'img/', batch_size)

    outputs_batch = model(inputs_batch)
    guess_coords_batch = [util.output_to_joints(output, img_batch[i].shape) for i, output in enumerate(outputs_batch)]

    for i, img in enumerate(img_batch):
        util.draw_joints(img, gt_coords_batch[i], (255,0,0))
        util.draw_joints(img, guess_coords_batch[i], (0,0,255))
        cv2.imshow('test', img)
        cv2.waitKey(0)

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

val_annotations = util.read_annotations(val_path + 'annotations.csv')
train_annotations = util.read_annotations(train_path + 'annotations.csv')

batch_size = 32

if args.weights_path:
    model.load_state_dict(torch.load(args.weights_path))
elif args.visualise or args.validate:
    raise Exception('Visualise / validate modes require pre-trained weights (use --weights_path option).')

if args.validate:
    validate()

elif args.visualise:
    visualise()

else:  # train
    train(args.epoch_count if args.epoch_count else 20, epoch_offset=(args.epoch_offset if args.epoch_offset else 1))
    annotations = util.read_annotations(train_path + 'annotations.csv')
    


