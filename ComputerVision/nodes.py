from PIL import Image, ImageOps
import numpy as np

# this is for the teachablemachine model
import tensorflow as tf
from keras.models import load_model
import cv2

from io import BytesIO
from server import PromptServer, BinaryEventTypes


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
    # OUTPUT_NODE = True
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
                "classA": ("STRING", {"default": "class"}),
                "outputA": ("CLASSIFICATIONS", {"forceInput":True}),
                "classB": ("STRING", {"default": "class"}),
                "outputB": ("CLASSIFICATIONS", {"forceInput":True}),
                "classC": ("STRING", {"default": "class"}),
                "outputC": ("CLASSIFICATIONS", {"forceInput":True}),
                "classD": ("STRING", {"default": "class"}),
                "outputD": ("CLASSIFICATIONS", {"forceInput":True}),
                "classE": ("STRING", {"default": "class"}),
                "outputE": ("CLASSIFICATIONS", {"forceInput":True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "combine_send_results"
    OUTPUT_NODE = True
    CATEGORY = "🐑 does_custom_nodes/Classification"

    def combine_send_results(self, classA, outputA, classB, outputB, classC, outputC, classD, outputD, classE, outputE):
        server = PromptServer.instance

        # Send the classifications to the server
        server.send_sync(
            "classification",
            { 
                classA: outputA,  # Dynamically constructed list
                classB: outputB,  # Dynamically constructed list
                classC: outputC,  # Dynamically constructed list
                classD: outputD,  # Dynamically constructed list
                classE: outputE
            },
            server.client_id,
        )

        return {}

class SwitchClassifiation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("CLASSIFICATIONS", {"forceInput":True}),
                "input2": ("CLASSIFICATIONS", {"forceInput":True}),
                "maskIsEmpty": ("BOOLEAN", {"defaultInput":True})
            },
        }
    
    RETURN_TYPES = ("CLASSIFICATIONS",)
    FUNCTION = "switch_classification"
    CATEGORY = "🐑 does_custom_nodes/Classification"

    def switch_classification(self, input1, input2, maskIsEmpty):
        return (input1 if maskIsEmpty else input2,)
    
class StringToClassification:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "classNameA": ("STRING", {"default": "data here"}),
                "confidenceA": ("INT", {"min": 0, "max": 1}),
                "classNameB": ("STRING", {"default": "data here"}),
                "confidenceB": ("INT", {"min": 0, "max": 1}),
            }
        }
    
    RETURN_TYPES = ("CLASSIFICATIONS",)
    FUNCTION = "string_to_classification"
    CATEGORY = "🐑 does_custom_nodes/Classification"

    def string_to_classification(self, classNameA, confidenceA, classNameB, confidenceB):
        result = []
        result.append({"className": classNameA, "confidence": f"{confidenceA:.2f}"})
        result.append({"className": classNameB, "confidence": f"{confidenceB:.2f}"})

        results = []
        results.append(result)
        return results
