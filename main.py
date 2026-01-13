from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import streamlit as st



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
st.title('CIFAR10 AI model')
st.text('Upload image with a number, and model will recognize it')

file = st.file_uploader('Choose of drop an image', type=['svg', 'png', 'jpg', 'jpeg'])

if not file:
    st.warning('No file is uploaded')
else:
    st.image(file, caption='Uploaded image')
    if st.button('Recognize the image'):
        try:
            image_data = file.read()
            if not image_data:
                raise HTTPException(status_code=400, detail='No image is given')

            img = Image.open(io.BytesIO(image_data)).convert("RGB")  # ✅ Конвертация в RGB
            img_tensor = train_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(img_tensor)
                pred = y_pred.argmax(dim=1).item()

            st.success({"Answer": class_names[pred]})

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


