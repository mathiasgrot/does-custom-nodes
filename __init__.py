from . import api, nodes

NODE_CLASS_MAPPINGS = {
    "DOES_custom_nodes_SendImageTypeWebSocket": nodes.SendImageTypeWebSocket,
    "DOES_custom_nodes_SendStatusMessageWebSocket": nodes.SendStatusMessageWebSocket,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DOES_custom_nodes_SendImageTypeWebSocket": "Send Image and Type(WebSocket, Base64)",
    "DOES_custom_nodes_SendStatusMessageWebSocket": "Send Message and Type(WebSocket)",
}