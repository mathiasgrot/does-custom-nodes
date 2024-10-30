from PIL import Image
import numpy as np
import base64
import torch
from io import BytesIO
from server import PromptServer, BinaryEventTypes


#this function is added
class SendImageTypeWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["PNG", "JPEG"], {"default": "PNG"}),
                "text":  ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_images_type"
    OUTPUT_NODE = True
    CATEGORY = "does_custom_nodes"

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
    
#this function is added
class SendStatusMessageWebSocket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "text":  ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_message"
    OUTPUT_NODE = True
    CATEGORY = "does_custom_nodes"

    def send_message(self, images, text):
        #self.prompt_queue.get_tasks_remaining()
        server = PromptServer.instance
        server.send_sync(
            "prompt queued",
            { "type": text},
            server.client_id,
        )
        return {}