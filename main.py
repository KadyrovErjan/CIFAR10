from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image



train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class CheckImage2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.first2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # узнаём выходной размер автоматически
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)  # поменяй 32x32 на свой размер входа
            out = self.first2(dummy)
            out_features = out.view(1, -1).size(1)

        self.second2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.first2(x)
        x = self.second2(x)
        return x



check_image_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage2()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()


class_names = [
     'airplane',
     'automobile',
     'bird',
     'cat',
     'deer',
     'dog',
     'frog',
     'horse',
     'ship',
     'truck'
]
@check_image_app.post("/predict/")
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Файл кошулган жок")

        img = Image.open(io.BytesIO(image_data))
        img_tensor = train_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {"Answer": class_names[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(check_image_app, host='127.0.0.1', port=8003)

