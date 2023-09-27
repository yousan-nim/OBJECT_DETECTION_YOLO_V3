import torch
import argparse
from tqdm import tqdm
from PIL import ImageFile
import torch.optim as optim

from .Model import YOLOv3, YOLO_LOSS
from .Transform import train_transform
from .Datasets import Dataset
from .Utills import (save_checkpoint,
                    ANCHORS, 
                    iou, 
                    s, 
                    convert_cells_to_bboxes, 
                    nms, 
                    plot_image, 
                    save_image)

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else "cpu"

def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors): 
    
    progress_bar = tqdm(loader, leave=True)
    
    losses = [] 
    
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )
        
        with torch.cuda.amp.autocast():
            outputs = model(x)
            loss = (
                    loss_fn(outputs[0], y0, scaled_anchors[0]) + 
                    loss_fn(outputs[1], y1, scaled_anchors[1]) + 
                    loss_fn(outputs[2], y2, scaled_anchors[2]) 
            )
        
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        mean_loss= sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)
        
        
def trainer(args):
    model = YOLOv3().to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    loss_fn = YOLO_LOSS()
    scaler = torch.cuda.amp.GradScaler()
    train_dataset = Dataset(
        csv_file=args.csv_file,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        anchors=ANCHORS,
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        num_workers = 0,
        shuffle = True,
        pin_memory = True,
    )
    scaled_anchors = (
        torch.tensor(ANCHORS) * 
        torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(device)

    for e in range(1, args.epochs+1):
        print("Epoch:", e)
        training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if args.save_model:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
            
    
    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--load_model', default=True)
    parser.add_argument('--save_model', default=True)
    parser.add_argument('--checkpoint', default='./checkpoint.pth.tar')
    parser.add_argument('--csv_file',   default="./train.csv")
    parser.add_argument('--image_dir',  default="./images")
    parser.add_argument('--label_dir',  default="./labels")
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--epochs', default=50)

    args = parser.parse_args()
    
    trainer(args)
    
    print("Training Finished...")