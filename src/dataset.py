# https://huggingface.co/docs/datasets/quickstart

import torch

from datasets import Audio
from datasets import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def collate_fn(dataset):
    images = []
    labels = []
    for ds in dataset:
        images.append((ds["pixel_values"]))
        labels.append(ds["labels"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}


def preprocess_function(dataset):
    return feature_extractor(
        [ds["array"] for ds in dataset["audio"]],
        sampling_rate=16000,
        padding=True,
        max_length=100000,
        truncation=True,
    )


def transforms(dataset):
    jitter = Compose([ColorJitter(brightness=0.5, hue=0.5), ToTensor()])
    dataset["pixel_values"] = [jitter(image.convert("RGB")) for image in dataset["image"]]
    return dataset


def encode(dataset):
    return tokenizer(dataset["sentence1"], dataset["sentence2"], truncation=True, padding="max_length")


# Load the model and feature extractor
model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Load the dataset
audio_dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
# Resample audio to match the sampling rate of facebook/wav2vec2-base-960h it was trained on
print("Sampling rate before:", audio_dataset[0]["audio"]["sampling_rate"])
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))
print("Sampling rate after:", audio_dataset[0]["audio"]["sampling_rate"])

# Preprocess the dataset
audio_dataset = audio_dataset.map(preprocess_function, batched=True)
audio_dataset = audio_dataset.rename_column("intent_class", "labels")
audio_dataset.set_format(type="torch", columns=["input_values", "labels"])

vision_dataset = load_dataset("beans", split="train")
vision_dataset = vision_dataset.cast_column("image", Image(mode="RGB"))
vision_dataset = vision_dataset.with_transform(transforms)

nlp_dataset = load_dataset("glue", "mrpc", split="train")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

nlp_dataset = nlp_dataset.map(encode, batched=True)
nlp_dataset = nlp_dataset.map(lambda data: {"labels": data["label"]}, batched=True)
nlp_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# dataloader = DataLoader(audio_dataset, batch_size=4)
# dataloader = DataLoader(vision_dataset, collate_fn=collate_fn, batch_size=4)
# dataloader = torch.utils.data.DataLoader(nlp_dataset, batch_size=32)
