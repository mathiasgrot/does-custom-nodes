from PIL import Image, ImageOps
import numpy as np
import torch
from io import BytesIO


class IsMaskEmpty:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ["BOOLEAN"]
    RETURN_NAMES = ["boolean"]

    FUNCTION = "main"
    CATEGORY = "🐑 does_custom_nodes/Utils"

    def main(self, mask):
        return (torch.all(mask == 0).item(),)


class CombineImagesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "background": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"
    OUTPUT_NODE = True
    CATEGORY = "🐑 does_custom_nodes/Utils"

    def combine(self, source, background, mask):
        # Convert tensors to NumPy arrays
        source_np = source.cpu().numpy()
        background_np = background.cpu().numpy()

        # Convert the source, mask, and background tensors to PIL images (assuming [H, W, C] format)
        source_pil = Image.fromarray((source_np * 255).astype(np.uint8))  # scale to [0, 255]
        background_pil = Image.fromarray((background_np * 255).astype(np.uint8))  # scale to [0, 255]

        # Handle mask (if mask is not already of shape [B, H, W, C], adjust it)
        if len(mask.shape) == 3:  # If mask is of shape [B, H, W], squeeze to [H, W]
            mask = mask.squeeze(0)

        # Ensure the mask is a 2D tensor [H, W]
        if len(mask.shape) == 2:
            mask_pil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))  # Convert to 0-255 for PIL
        else:
            # Handle any other case where mask may have extra dimensions (like [B, H, W, C])
            mask_pil = Image.fromarray((mask[0].numpy() * 255).astype(np.uint8))  # Using the first batch element if applicable

        # Resize source, mask, and background to fit within the background, maintaining aspect ratio
        bg_width, bg_height = background_pil.size
        src_width, src_height = source_pil.size
        
        # Calculate scale factor to fit source within background
        scale_factor = min(bg_width / src_width, bg_height / src_height)
        new_width = int(src_width * scale_factor)
        new_height = int(src_height * scale_factor)
        
        source_resized = source_pil.resize((new_width, new_height), Image.ANTIALIAS)
        mask_resized = mask_pil.resize((new_width, new_height), Image.ANTIALIAS)
        
        # Calculate position to center the source on the background
        x_offset = (bg_width - new_width) // 2
        y_offset = (bg_height - new_height) // 2
        
        # Create a copy of the background to paste on
        combined = background_pil.copy()
        
        # Paste the resized source onto the background using the mask
        combined.paste(source_resized, (x_offset, y_offset), mask_resized)
        
        # Return the final image
        return combined