from . import nodes

NODE_CLASS_MAPPINGS = {
    "DOES_custom_nodes_SendImageTypeWebSocket": nodes.SendImageTypeWebSocket,
    "DOES_custom_nodes_SendStatusMessageWebSocket": nodes.SendStatusMessageWebSocket,
    "DOES_custom_nodes_SendImageWithMessageSocket": nodes.SendImageWithMessageWebSocket,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DOES_custom_nodes_SendImageTypeWebSocket": "🐑 Send Image and Type(WebSocket, Base64)",
    "DOES_custom_nodes_SendStatusMessageWebSocket": "🐑 Send Status Message(WebSocket)",
    "DOES_custom_nodes_SendImageWithMessageSocket": "🐑 Send Image with Message(WebSocket, Base64)",
}