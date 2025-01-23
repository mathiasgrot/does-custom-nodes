
import numpy as np
import torch
from torchvision import transforms
from datasets import load_dataset, Dataset, Image


dataset_name_list = [
    {"name":"conorcl/portraits-512", "column": "image"},
    {"name": "lcolok/Asian_Regularization_images", "column": "image"},
    {"name": "Kalva014/male-asian-hairstyles", "column": "image"},
    {"name": "conorcl/portraits", "column": "image"},
    {"name": "conorcl/portraits3", "column": "image"},
    {"name": "gaunernst/flux-dev-portrait", "column": "webp"},
    {"name": "strangerzonehf/Flux-Generated-Super-Portrait", "column": "image"},
    {"name": "zackli4ai/lora-portrait-test", "column": "image"},
    {"name": "yuvalkirstain/portrait_dreambooth", "column": "image"},
    {"name": "sugarquark/flux-portrait", "column": "image"},
    {"name": "creampietzuzyu99x/facesets", "column": "image"},
]

# dataset_name_list = [
#     "conorcl/portraits-512",
#     "lcolok/Asian_Regularization_images",
#     "Kalva014/male-asian-hairstyles",
#     "conorcl/portraits",
#     "conorcl/portraits3",
#     "gaunernst/flux-dev-portrait",
#     "strangerzonehf/Flux-Generated-Super-Portrait",
#     "zackli4ai/lora-portrait-test",
#     "yuvalkirstain/portrait_dreambooth",
#     "sugarquark/flux-portrait",
#     "creampietzuzyu99x/facesets"
# ]

class LoadDataSet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset": ([entry["name"] for entry in dataset_name_list],  # Extract names
                            {"default": dataset_name_list[0]["name"]}),
                "useStreaming": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("DATASET", "ITERATOR")
    FUNCTION = "load_dataset"
    OUTPUT_NODE = False
    CATEGORY = "🐑 does_custom_nodes/Utils"

    def load_dataset(self, dataset, useStreaming):
        ds = load_dataset(dataset, split="train", streaming=useStreaming)

        iterator = None  # No iterator needed for non-streaming mode

        if useStreaming:
            iterator = iter(ds)  # Create and store dataset iterator

        return (ds, iterator)  # Return dataset reference + streaming flag
    
class LoadDataSetFromURL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset": ("STRING", {"default": "namespace/your_dataset_name"}),
                "useStreaming": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("DATASET", "ITERATOR")  # Also return streaming flag
    FUNCTION = "load_dataset_from_url"
    OUTPUT_NODE = False
    CATEGORY = "🐑 does_custom_nodes/Utils"

    def load_dataset_from_url(self, dataset, useStreaming):
        """Loads the dataset (with optional streaming mode)."""
        ds = load_dataset(dataset, split="train", streaming=useStreaming)
        iterator = None  # No iterator needed for non-streaming mode

        if useStreaming:
            iterator = iter(ds)  # Create and store dataset iterator

        return (ds, iterator)  # Return dataset reference + streaming flag


class LoadNextImage:
    """Returns images one by one from a loaded dataset."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
            },
            "optional": {
                "iterator": ("ITERATOR", {"forceInput": True}),  # Accept iterator
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")  # Also return next index for iteration
    FUNCTION = "load_next_image"
    OUTPUT_NODE = False
    CATEGORY = "🐑 does_custom_nodes/Utils"

    def load_next_image(self, dataset, iterator, index):
        """Fetches the next image efficiently."""

        # If using streaming mode, convert dataset to an iterator
        if iterator is not None:
            image = next(iterator)["image"]
        else:
            image = dataset[index]["image"]

        # Convert image to a NumPy array and normalize to [0,1]
        # Define the transformation to convert a PIL image to a tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor()  # Converts PIL image to torch tensor and scales values to [0, 1]
        ])

        # Convert the PIL image to a torch.Tensor
        image_tensor = transform(image.convert("RGB")).float()
        image_tensor = image_tensor.permute(1, 2, 0).unsqueeze(0)  # [H, W, C] → [1, H, W, C]

        return (image_tensor, index)  # Return both tensor and next index