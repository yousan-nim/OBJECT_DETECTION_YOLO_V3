import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .Utills import load_checkpoint, ANCHORS, iou 


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, image_dir, label_dir, anchors,
        image_size=416, grid_sizes=[13, 26, 52],
        num_classes=20, transform=None
        ):
        self.label_list = pd.read_csv(csv_file)
        self.image_dir  = image_dir
        self.label_dir  = label_dir 
        self.image_size = image_size 
        self.transform = transform
        self.grid_sizes = grid_sizes
        self.anchors = torch.tensor(
            anchors[0] +  anchors[1] +  anchors[2] 
        )
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3 
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        
        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]
            
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6))
                   for s in self.grid_sizes]
        
        for box in bboxes: 
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices: 
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                
                s = self.grid_sizes[scale_idx] 
                
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]: 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = s* x - j, s * y - i 
                    
                    width_cell, height_cell = (width * s, height * s)
                    
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    
                    has_anchor[scale_idx] = True
                    
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:

                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
    
        return image, tuple(targets)