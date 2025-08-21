# deploy_model_gpu.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import uvicorn


# -----------------------------
# Define the PyTorch Model
# -----------------------------
class MyModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=1):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# -----------------------------
# Initialize device & model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel()
# Load your trained weights
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="GPU ML Model Deployment")


class InputData(BaseModel):
    features: list  # List of floats representing your input features


@app.post("/predict")
def predict(data: InputData):
    # Convert input to torch tensor and move to GPU
    x = torch.tensor([data.features], dtype=torch.float32).to(device)
    with torch.no_grad():
        y = model(x)
    return {"prediction": y.cpu().numpy().tolist()}


# -----------------------------
# Run the API (for local testing)
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("deploy_model_gpu:app", host="0.0.0.0", port=8080, reload=True)
