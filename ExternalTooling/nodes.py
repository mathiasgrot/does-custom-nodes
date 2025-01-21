from PIL import Image, ImageOps
import numpy as np
import base64

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
    CATEGORY = "🐑 does_custom_nodes/External_tools"

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
                "messageType":  (
                    ["prompt queued", "status", "colorData", "classification"], {"default": "status",},
                ),
                "data":  ("STRING", {"multiline": False, "default": "Message"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_message"
    OUTPUT_NODE = True
    CATEGORY = "🐑 does_custom_nodes/External_tools"

    def send_message(self, messageType, data):
        server = PromptServer.instance
        queueInfo = server.prompt_queue.get_current_queue() #get the first item in the queue to read the prompt id
        server.send_sync(
            messageType,
            { "prompt_id":queueInfo[0][0][1], "type": data},
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
    CATEGORY = "🐑 does_custom_nodes/External_tools"

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