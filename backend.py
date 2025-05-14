import asyncio
import json
import base64
import io
from PIL import Image
import numpy as np
import websockets
import torch
from torchvision.transforms import functional as TF

# This is a placeholder for your custom diffusion model
# Replace this with your actual model implementation
class DummyDiffusionModel:
    def __init__(self):
        # Initialize your PyTorch model here
        # For example:
        # self.model = YourCustomModel.load_from_checkpoint("path/to/checkpoint")
        # self.model.eval()
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
        
        # This is just a placeholder. Replace with your actual model inference
        # For example:
        # with torch.no_grad():
        #     output = self.model(key_embedding)
        
        # For demonstration purposes, we're just creating a colored frame
        # based on the key pressed
        
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
            
        # Add text to the frame
        # In a real implementation, you'd use a library like cv2 for this
        # For the placeholder, we'll just create a simple pattern
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

# WebSocket server handler
async def game_server(websocket):
    # Initialize the diffusion model
    model = DummyDiffusionModel()
    
    print("Client connected")
    
    try:
        async for message in websocket:
            # Parse the incoming message
            data = json.loads(message)
            
            if data['type'] == 'keypress':
                key = data['key']
                print(f"Received keypress: {key}")
                
                # Generate a new frame using the diffusion model
                frame = model.generate_frame(key)
                
                # Convert the frame to base64 encoded PNG
                frame_base64 = tensor_to_base64(frame)
                
                # Send the frame back to the client
                await websocket.send(json.dumps({
                    'type': 'frame',
                    'frame': frame_base64
                }))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

# Start the WebSocket server
async def main():
    async with websockets.serve(
        game_server,
        "localhost",
        8765
    ) as server:
    
        print("WebSocket server started at ws://localhost:8765")
    
        # Keep the server running
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
