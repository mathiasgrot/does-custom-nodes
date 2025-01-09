from .ComputerVision import nodes as cv_nodes
from .ExternalTooling import nodes as tooling_nodes
from .Utils import nodes as util_nodes

NODE_CLASS_MAPPINGS = {
    "DOES_custom_nodes_SendImageTypeWebSocket": tooling_nodes.SendImageTypeWebSocket,
    "DOES_custom_nodes_SendStatusMessageWebSocket": tooling_nodes.SendStatusMessageWebSocket,
    "DOES_custom_nodes_SendImageWithMessageSocket": tooling_nodes.SendImageWithMessageWebSocket,
    "DOES_custom_nodes_TeachableMachine": cv_nodes.TeachableMachine,
    "DOES_custom_nodes_SwitchClassifiation": cv_nodes.SwitchClassifiation,
    "DOES_custom_nodes_StringToClassification": cv_nodes.StringToClassification,
    "DOES_custom_nodes_CombineClassificationResults": cv_nodes.CombineClassificationResults,
    "DOES_custom_nodes_CombineImagesNode": util_nodes.CombineImagesNode,
    "DOES_custom_nodes_IsMaskEmpty": util_nodes.IsMaskEmpty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DOES_custom_nodes_SendImageTypeWebSocket": "🐑 Send Image and Type(WebSocket, Base64)",
    "DOES_custom_nodes_SendStatusMessageWebSocket": "🐑 Send Status Message(WebSocket)",
    "DOES_custom_nodes_SendImageWithMessageSocket": "🐑 Send Image with Message(WebSocket, Base64)",
    "DOES_custom_nodes_TeachableMachine": "🐑 Classify image (🍁 Maplestory Avatar)",
    "DOES_custom_nodes_CombineImagesNode": "🐑 Combine images",
    "DOES_custom_nodes_IsMaskEmpty": "🐑 Is mask empty",
    "DOES_custom_nodes_SwitchClassifiation": "🐑 Switch Classification",
    "DOES_custom_nodes_StringToClassification": "🐑 String to Classification",
    "DOES_custom_nodes_CombineClassificationResults": "🐑 Combine and Send Classifications",
}