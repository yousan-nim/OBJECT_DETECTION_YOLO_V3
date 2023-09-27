import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .Utills import image_size

train_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
        ),
        
        A.ColorJitter(
            brightness=0.5, contrast=0.5,
            saturation=0.5, hue=0.5, p=0.5
        ),
        
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ),
        
        ToTensorV2()
    ], 
    bbox_params=A.BboxParams(
                    format="yolo", 
                    min_visibility=0.4, 
                    label_fields=[]
                )
)
  

test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
        ),

        A.Normalize(
            mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
        ),

        ToTensorV2()
    ],
    bbox_params=A.BboxParams(
                    format="yolo", 
                    min_visibility=0.4, 
                    label_fields=[]
                )
)