from torchvision import transforms as T
from scipy.stats import multivariate_normal as mv_norm
from PIL import Image
import numpy as np
import torch
import cv2
import csv
import math

JOINT_LABEL_DICT = {
    'R_Ankle': 0,
    'R_Knee': 1,
    'R_Hip': 2,
    'L_Hip': 3,
    'L_Knee': 4,
    'L_Ankle': 5,
    'B_Pelvis': 6,
    'B_Spine': 7,
    'B_Neck': 8,
    'B_Head': 9,
    'R_Wrist': 10,
    'R_Elbow': 11,
    'R_Shoulder': 12,
    'L_Shoulder': 13,
    'L_Elbow': 14,
    'L_Wrist': 15,
}

def load_img(path, img_id):
    img = cv2.imread(path + img_id, cv2.COLOR_BGR2RGB)  # read image as np ndarray
    if len(img.shape) == 2:  # convert grayscale to rgb
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def img_to_input(img, input_dims=(256, 256)):
    # convert np array to tensor and normalize + resize

    pil_img = Image.fromarray(img)
    transform = T.Compose([
        T.Resize(input_dims),
        T.ToTensor(),
        T.Normalize(
            # mean and std values according to torchvision docs
            # https://pytorch.org/docs/stable/torchvision/models.html
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    return transform(pil_img)

def read_annotations(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # create dict using img id (filename) as key with list of joints as value
    return {row[0]:row[1:] for row in rows}

def parse_annotation(annotation):

    '''
        Converts csv annotation to joint coordinate list representation.
    '''
    
    return [
        [int(annotation[i_*3]), int(annotation[i_*3+1])] 
        if annotation[i_*3] != 'nan' else [] 
        for i_ in range(16)
    ]

def output_to_joints(output, img_dim, conf_thresh=0):
    coords_list = []
    for joint_i in range(len(JOINT_LABEL_DICT)):
        joint_coords = []  # empty list implies no joint detected

        heatmap = output[joint_i]
        max_i = torch.argmax(heatmap)
        max_coords = (max_i % heatmap.shape[1], max_i // heatmap.shape[1])

        # we ignore joints where the model's confidence is insufficient
        if heatmap[max_coords].item() >= conf_thresh:
            # convert output coords to img coords
            joint_coords = [
                max_coords[0] * img_dim[1] / heatmap.shape[1],
                max_coords[1] * img_dim[0] / heatmap.shape[0]
            ]

        coords_list.append(joint_coords)

    return coords_list

def draw_joints(img, coords_list, colour=(255,0,0)):
    for joint_coords in coords_list:
        if joint_coords != []:  # joint detected
            cv2.ellipse(img, (joint_coords[0], joint_coords[1]), (3,3), 0, 0, 360, colour, -1)
        else:  # joint not detected, so don't draw anything
            pass

def generate_gt_heatmaps(gt_coords_list, img_dims, heatmap_dims):

    '''
        Generates the ground truth (target) heat maps to be used by the model
        during training. This is done by applying a gaussian distribution centred on
        each joint in turn, with identity covariance.
    '''

    heatmaps = []
    for gt_coords in gt_coords_list:
        # convert image coords to heatmap coords

        heatmap = torch.zeros(heatmap_dims)
        if gt_coords == []:  # joint not present in image so we skip
            pass
            
        else:
            heatmap_coords = [
                # kind of confusing syntax since dimensions are rows, cols but
                # coords are x, y (col, row)
                min(max(int(gt_coords[0] * heatmap_dims[1] / img_dims[1]), 0), 63), # constrain to valid indices as sometimes the gt coords
                min(max(int(gt_coords[1] * heatmap_dims[0] / img_dims[0]), 0), 63)  # can fall slightly outside image bounds
            ]
            dist = mv_norm(mean=heatmap_coords, cov=[[1,0],[0,1]])  # create gaussian distribution
            factor = 1 / dist.pdf([heatmap_coords])  #  scale factor so centre prob. dens. is 1

            # apply 2d gaussian centred on ground truth joint coord
            # (only need to do small area surrounding as prob. dens. falls off rapidly) 
            for y in range(max(heatmap_coords[1] - 4, 0), min(heatmap_coords[1] + 4, 63)):
                for x in range(max(heatmap_coords[0] - 4, 0), min(heatmap_coords[0] + 4, 63)):
                    # pdf call is using x,y rather than y,x as mean was defined using x,y coords
                    heatmap[y, x] = dist.pdf([x, y]) * factor
      

        heatmaps.append(heatmap)
    
    return torch.stack(heatmaps, dim=0)