import time
import torch

from datasets import Audio
from datasets import load_dataset
from transformers import pipeline


# Enable TensorFloat-32 for Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")

# 0 for GPU, -1 for CPU
device = 0 if torch.cuda.is_available() else -1

# Load the dataset
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

# Load the speech recognizer
speech_recognizer = pipeline(
    "automatic-speech-recognition",
    device=device,
    model_kwargs={"torch_dtype": torch.float32},
    trust_remote_code=True,
)

# Cast the audio column to the Audio type
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

# Time the classification
start_time = time.time()
result = speech_recognizer(dataset[:3]["audio"])
end_time = time.time()
total_time = end_time - start_time

print(f"\nUsing {'GPU' if device == 0 else 'CPU'}")
print(f"Total time: {total_time:.2f} seconds")
print([d["text"] for d in result])
