from PIL import Image
import numpy as np
import base64
import torch
from io import BytesIO
from server import PromptServer, BinaryEventTypes

def encodeImageToBase64(image, format):
    array = 255.0 * tensor.cpu().numpy()
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    # Convert image to a Base64-encoded string
    buffer = BytesIO()
    image.save(buffer, format=format)  # Save the image in the desired format
    image_data = buffer.getvalue()
    return base64.b64encode(image_data).decode('utf-8')  # Encode to Base64



#this function is added
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
            # array = 255.0 * tensor.cpu().numpy()
            # image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            # # Convert image to a Base64-encoded string
            # buffer = BytesIO()
            # image.save(buffer, format=format)  # Save the image in the desired format
            # image_data = buffer.getvalue()
            # base64_image = base64.b64encode(image_data).decode('utf-8')  # Encode to Base64
            base64_image = encodeImageToBase64(images, format)

            server = PromptServer.instance
            server.send_sync(
                "base64Image",
                { "type": type, "base64Image": base64_image},
                server.client_id,
            )

            # results.append(
            #     {"source": "websocket", "content-type": f"image/{format.lower()}", "type": "output"}
            # )

        return {}
    
#this function is added
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
        queuInfo = server.prompt_queue.get_current_queue() #get the first item in the queue to read the prompt id
        server.send_sync(
            "prompt queued",
            { "prompt_id":queuInfo[0][0][1], "type": text},
            server.client_id,
        )
        return {}
    
    #this function is added
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
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            # Convert image to a Base64-encoded string
            buffer = BytesIO()
            image.save(buffer, format=format)  # Save the image in the desired format
            image_data = buffer.getvalue()
            base64_image = base64.b64encode(image_data).decode('utf-8')  # Encode to Base64

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