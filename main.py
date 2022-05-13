import torch
from datasets import load_dataset

from descry import VisionTransformer

vt = VisionTransformer()
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
vt.forward(image)
