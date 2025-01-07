from PIL import Image, ImageOps
import numpy as np
import base64
import torch

# this is for the teachablemachine model
import tensorflow as tf
from keras.models import load_model
import cv2

from io import BytesIO
from server import PromptServer, BinaryEventTypes


def encodeImageToBase64(tensor, format):
    array = 255.0 * tensor.cpu().numpy()
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    # Convert image to a Base64-encoded string
    buffer = BytesIO()
    image.save(buffer, format=format)  # Save the image in the desired format
    image_data = buffer.getvalue()
    
    return base64.b64encode(image_data).decode('utf-8')  # Encode to Base64



class SendImageTypeWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "JPEG"], {"default": "PNG"}),
                "type": (["Diffuse", "Depth", "Normal", "Roughness"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_images_type"
    OUTPUT_NODE = True
    CATEGORY = "🐑 does_custom_nodes"

    def send_images_type(self, images, format, type):
        results = []
        for tensor in images:
            base64_image = encodeImageToBase64(tensor, format)

            server = PromptServer.instance
            server.send_sync(
                "base64Image",
                { "type": type, "base64Image": base64_image},
                server.client_id,
            )
        return {}
    

class SendStatusMessageWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "text":  ("STRING", {"multiline": False, "default": "Message"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_message"
    OUTPUT_NODE = True
    CATEGORY = "🐑 does_custom_nodes"

    def send_message(self, images, text):
        server = PromptServer.instance
        queueInfo = server.prompt_queue.get_current_queue() #get the first item in the queue to read the prompt id
        server.send_sync(
            "prompt queued",
            { "prompt_id":queueInfo[0][0][1], "type": text},
            server.client_id,
        )
        return {}


class SendImageWithMessageWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "JPEG"], {"default": "PNG"}),
                "text":  ("STRING", {"multiline": True, "default": "Diffuse"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_images_type"
    OUTPUT_NODE = True
    CATEGORY = "🐑 does_custom_nodes"

    def send_images_type(self, images, format, text):
        results = []
        for tensor in images:
            base64_image = encodeImageToBase64(tensor, format)

            server = PromptServer.instance
            server.send_sync(
                "base64Image",
                { "type": text, "base64Image": base64_image},
                server.client_id,
            )

            # results.append(
            #     {"source": "websocket", "content-type": f"image/{format.lower()}", "type": "output"}
            # )

        return {}
    

# had to use older version of tensorflow==2.12.1 
# installed like this inside the comfy ui python environment
# python_embeded\python.exe -m pip install tensorflow==2.12.1
class TeachableMachine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_path": ("STRING", {"default": "path/to/keras_Model.h5"}),
                "labels_path": ("STRING", {"default": "path/to/labels.txt"})
            }
        }

    RETURN_TYPES = ("CLASSIFICATIONS",)
    FUNCTION = "classify_images"
    OUTPUT_NODE = True
    CATEGORY = "🐑 does_custom_nodes/Classification"

    def classify_images(self, images, model_path, labels_path):
        # server = PromptServer.instance
    
        # Load the model
        model = load_model(model_path, compile=False)

        # Load class names
        with open(labels_path, "r") as f:
            class_names = f.readlines()

        results = []

        for image_tensor in images:
            # Preprocess image tensor (assuming it is in a [0, 1] range)
            image_np = np.array(image_tensor * 255, dtype=np.float32)
            image_np = cv2.resize(image_np, (224, 224), interpolation=cv2.INTER_AREA)
            image_np = np.expand_dims((image_np / 127.5) - 1, axis=0)

            # Predict class
            prediction = model.predict(image_np)

            # get the highest class
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]

            # Dynamically create the classification list based on the number of classes
            result = []
            for i, score in enumerate(prediction[0]):
                class_name = class_names[i].strip()
                # Remove the number and space from the class name (e.g., '0 hair_long' -> 'hair_long')
                class_name = class_name.split(' ', 1)[-1]
                result.append({"className": class_name, "confidence": f"{score:.2f}"})

            # Send the classifications to the server
            # server.send_sync(
            #     "classification",
            #     { 
            #         "classifications": result  # Dynamically constructed list
            #     },
            #     server.client_id,
            # )

            results.append(result)

            # for res in result:
            #     print(res)    

        return results
    
class CombineClassificationResults:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "outputA": ("CLASSIFICATIONS", {"forceInput":True}),
                "outputB": ("CLASSIFICATIONS", {"forceInput":True}),
                "outputC": ("CLASSIFICATIONS", {"forceInput":True}),
                "outputD": ("STRING",{"default": "type A"}),
                "titleA": ("STRING", {"default": "type A"}),
                "titleB": ("STRING", {"default": "type B"}),
                "titleC": ("STRING", {"default": "type B"}),
                "titleD": ("STRING", {"default": "type C"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "combine_send_results"
    OUTPUT_NODE = True
    CATEGORY = "🐑 does_custom_nodes/Classification"

    def combine_send_results(self, outputA, outputB, outputC, outputD, titleA, titleB, titleC, titleD):
        server = PromptServer.instance

        # Send the classifications to the server
        server.send_sync(
            "classification",
            { 
                titleA: outputA,  # Dynamically constructed list
                titleB: outputB,  # Dynamically constructed list
                titleC: outputC,  # Dynamically constructed list
                titleD: outputD # temporary moondream output
            },
            server.client_id,
        )

        return {}

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
    CATEGORY = "🐑 does_custom_nodes"

    def main(self, mask):
        return (torch.all(mask == 0).item(),)
    

class SwitchClassifiation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("STRING", {"default": "class"}),
                "input2": ("CLASSIFICATIONS", {"forceInput":True}),
                "maskIsEmpty": ("BOOLEAN",)
            },
        }
    
    RETURN_TYPES = ("CLASSIFICATIONS",)
    FUNCTION = "switch_classification"
    CATEGORY = "🐑 does_custom_nodes/Classification"

    def switch_classification(self, input1, input2, skipOutput):
        result = []
        if skipOutput:
            result.append({"className": input1, "confidence": f"{0:.2f}"})
        return (result if skipOutput else input2,)

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
    CATEGORY = "🐑 does_custom_nodes"

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