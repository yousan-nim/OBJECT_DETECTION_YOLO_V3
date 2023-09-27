import argparse
import torch
import torch.optim as optim
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from .Model import YOLOv3, YOLO_LOSS
from .Transform import test_transform
from .Datasets import Dataset
from .Utills import (s,
                    nms, 
                    ANCHORS, 
                    load_checkpoint, 
                    convert_cells_to_bboxes, 
                    plot_image, save_image)
                    

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(args):
    model = YOLOv3().to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    loss_fn = YOLO_LOSS()
    scaler = torch.cuda.amp.GradScaler()

    if args.load_model: 
        load_checkpoint(args.checkpoint, model, optimizer, args.lr)

    test_dataset = Dataset(
        csv_file=args.csv_file,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        anchors=ANCHORS,
        transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
    )

    x, y = next(iter(test_loader))
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)

        bboxes = [[] for _ in range(x.shape[0])]
        anchors = (
            torch.tensor(ANCHORS)
                * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
            ).to(device)

        for i in range(3):
            batch_size, A, S, _, _ = output[i].shape
            anchor = anchors[i]
            boxes_scale_i = convert_cells_to_bboxes(
                output[i], anchor, s=S, is_predictions=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()
        for i in range(batch_size):
            nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.9)
            
            plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
            
            save_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, i)
            
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--load_model', default=True)
    parser.add_argument('--checkpoint', default='./checkpoint.pth.tar')
    parser.add_argument('--csv_file',   default="./test.csv")
    parser.add_argument('--image_dir',  default="./images")
    parser.add_argument('--label_dir',  default="./labels")
    parser.add_argument('--batch_size', default=1)
    
    args = parser.parse_args()

    evaluate(args)