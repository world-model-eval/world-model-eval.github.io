import json
import base64
import io
import threading
from PIL import Image
import numpy as np
import torch
from simple_websocket_server import WebSocketServer, WebSocket

# This is a placeholder for your custom diffusion model
class DummyDiffusionModel:
    def __init__(self):
        # Initialize your PyTorch model here
        print("Initializing diffusion model...")
        
    def generate_frame(self, key_input):
        """
        Generate a new frame based on the keypress input
        
        Args:
            key_input (str): The key that was pressed
        
        Returns:
            torch.Tensor: Generated image tensor (H, W, C)
        """
        print(f"Generating frame for key: {key_input}")
        
        # Create a simple colored frame based on the ASCII value of the key
        height, width = 512, 512
        ascii_val = ord(key_input[0]) if key_input else 0
        
        r = (ascii_val * 13) % 256
        g = (ascii_val * 29) % 256
        b = (ascii_val * 71) % 256
        
        # Create a tensor with shape (H, W, 3)
        frame = torch.zeros((height, width, 3), dtype=torch.uint8)
        frame[:, :, 0] = r
        frame[:, :, 1] = g
        frame[:, :, 2] = b
        
        # Add some visual elements to the frame
        for i in range(10):
            pos = (ascii_val * (i+1)) % min(height, width)
            thickness = (i + 1) * 5
            frame[pos:pos+thickness, :, :] = 255 - frame[pos, 0, :]
            frame[:, pos:pos+thickness, :] = 255 - frame[0, pos, :]
            
        # Add text representation to the frame
        key_text = f"Key: {key_input}"
        for c_idx, c in enumerate(key_text):
            x = 100 + c_idx * 20
            y = 100
            if x < width and y < height:
                frame[y-10:y+10, x-10:x+10, :] = 255
                
        return frame

# Convert tensor to base64 encoded PNG
def tensor_to_base64(tensor):
    """Convert a PyTorch tensor to base64 encoded PNG"""
    # Convert to numpy
    if tensor.device != 'cpu':
        tensor = tensor.cpu()
    
    # Ensure tensor has the right shape and type for an image
    img_array = tensor.numpy()
    
    # If tensor is in range [0, 1], convert to [0, 255]
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array.astype('uint8'))
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    
    # Convert to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_str

# WebSocket Handler class
class GameServer(WebSocket):
    def __init__(self, server, sock, address):
        super().__init__(server, sock, address)
        self.model = None
    
    def handle(self):
        # Parse the incoming message
        try:
            data = json.loads(self.data)
            
            if data['type'] == 'keypress':
                key = data['key']
                print(f"Received keypress: {key}")
                
                # Initialize the model if not already done
                if self.model is None:
                    self.model = DummyDiffusionModel()
                
                # Generate a new frame using the diffusion model
                frame = self.model.generate_frame(key)
                
                # Convert the frame to base64 encoded PNG
                frame_base64 = tensor_to_base64(frame)
                
                # Send the frame back to the client
                self.send_message(json.dumps({
                    'type': 'frame',
                    'frame': frame_base64
                }))
        except Exception as e:
            print(f"Error handling message: {e}")
    
    def connected(self):
        print(f"Client connected: {self.address}")
    
    def handle_close(self):
        print(f"Client disconnected: {self.address}")

# Start the WebSocket server
def start_server():
    server = WebSocketServer('localhost', 8765, GameServer)
    print("WebSocket server started at ws://localhost:8765")
    server.serve_forever()

if __name__ == "__main__":
    # Start the server in the main thread
    start_server()
