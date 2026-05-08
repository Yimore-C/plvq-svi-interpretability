import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import timm
import numpy as np
import argparse
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 1. Model definition
class ConvNeXtV2Regression(nn.Module):
    def __init__(self, model_name='convnextv2_tiny', pretrained=True):
        super(ConvNeXtV2Regression, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.regressor(self.backbone(x))

# 2. Dataset
class PLVQDataset(Dataset):
    def __init__(self, data_file, img_dir, transform=None, mode='train'):
        self.data = pd.read_excel(data_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['Image_name']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if self.mode == 'predict':
            return image, img_name
        else:
            label = row['Expert_Score'] 
            return image, torch.tensor(label, dtype=torch.float32)

# 3. Optimisation and training
def optimize_and_train(args, device):
    full_dataset = PLVQDataset(args.input_data, args.img_dir, 
                               transform=get_transform(), mode='train')
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        
        # cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        rmses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):
            if fold > 0: break
            train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=args.batch_size)

            model = ConvNeXtV2Regression().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()

            for epoch in range(args.epochs):
                model.train()
                for imgs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(imgs.to(device)), labels.to(device).view(-1, 1))
                    loss.backward()
                    optimizer.step()

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    out = model(imgs.to(device))
                    preds.extend(out.cpu().numpy())
                    targets.extend(labels.numpy())
            rmses.append(np.sqrt(mean_squared_error(targets, preds)))
        return np.mean(rmses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    print(f"Best params: {study.best_params}")

# 4. Prediction logic
def predict_only(args, device):
    print("Running in Prediction Mode...")
    model = ConvNeXtV2Regression(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    dataset = PLVQDataset(args.input_data, args.img_dir, 
                           transform=get_transform(), mode='predict')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = []
    with torch.no_grad():
        for image, name in loader:
            output = model(image.to(device))
            results.append({'Image_name': name[0], 'Predicted_Value': output.item()})
            if len(results) % 100 == 0: print(f"Processed {len(results)} images...")
    
    pd.DataFrame(results).to_excel(args.output_xlsx, index=False)
    print(f"Results saved to {args.output_xlsx}")

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['optimize', 'predict'], required=True)
    parser.add_argument('--img_dir', default='./data/images')
    parser.add_argument('--input_data', default='./data/data.xlsx')
    parser.add_argument('--model_path', default='./models/best_model.pth')
    parser.add_argument('--output_xlsx', default='./results/predictions.xlsx')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_trials', type=int, default=20)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'optimize':
        optimize_and_train(args, device)
    else:
        predict_only(args, device)
