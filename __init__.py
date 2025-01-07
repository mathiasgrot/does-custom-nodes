from . import nodes

NODE_CLASS_MAPPINGS = {
    "DOES_custom_nodes_SendImageTypeWebSocket": nodes.SendImageTypeWebSocket,
    "DOES_custom_nodes_SendStatusMessageWebSocket": nodes.SendStatusMessageWebSocket,
    "DOES_custom_nodes_SendImageWithMessageSocket": nodes.SendImageWithMessageWebSocket,
    "DOES_custom_nodes_TeachableMachine": nodes.TeachableMachine,
    "DOES_custom_nodes_CombineImagesNode": nodes.CombineImagesNode,
    "DOES_custom_nodes_IsMaskEmpty": nodes.IsMaskEmpty,
    "DOES_custom_nodes_SwitchClassifiation": nodes.SwitchClassifiation,
    "DOES_custom_nodes_CombineClassificationResults": nodes.CombineClassificationResults,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DOES_custom_nodes_SendImageTypeWebSocket": "🐑 Send Image and Type(WebSocket, Base64)",
    "DOES_custom_nodes_SendStatusMessageWebSocket": "🐑 Send Status Message(WebSocket)",
    "DOES_custom_nodes_SendImageWithMessageSocket": "🐑 Send Image with Message(WebSocket, Base64)",
    "DOES_custom_nodes_TeachableMachine": "🐑 Classify image (🍁 Maplestory Avatar)",
    "DOES_custom_nodes_CombineImagesNode": "🐑 Combine images",
    "DOES_custom_nodes_IsMaskEmpty": "🐑 Is mask empty",
    "DOES_custom_nodes_SwitchClassifiation": "🐑 Switch Classification",
    "DOES_custom_nodes_CombineClassificationResults": "🐑 Combine and Send Classifications",
}