import pickle
from tqdm import tqdm
import torch
import numpy as np
from algorithms.diffusion_forcing.df_video import DiffusionForcingVideo
from pytorch_memlab import profile
from torchvision.io import decode_image, write_video
import websockets
import base64
import io
import asyncio
import json
import time
import os
import random
import socket
import argparse
import logging
import uuid
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backend')

MODEL_DIR = os.getcwd()
CONFIG_PATH = f"{MODEL_DIR}/inference_config.pkl"
CKPT_PATH = f"{MODEL_DIR}/inference_ckpt.ckpt"
RGB_PATH = f"{MODEL_DIR}/app/images/xs_original_rgb_0.png"
APP_DIR = f"{MODEL_DIR}/app"

# Default coordinator server URL
COORDINATOR_URL = "ws://localhost:8701"

class WorldModel:
    """Handles the neural network model for video generation."""
    def __init__(self):
        self.device = "cuda:0"

        self.cfg = pickle.load(open(CONFIG_PATH, "rb")).algorithm
        self.cfg.use_server = False # disable original server implementation
        self.algo = DiffusionForcingVideo(self.cfg).to(self.device).eval()
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        self.algo.load_state_dict(ckpt["state_dict"])
        self.algo.cfg.scheduling_matrix = "pyramid"
        self.algo.chunk_size = 5

        self.reset(RGB_PATH)
        self.available_images = self._get_available_images()

    def _get_available_images(self):
        """Find all available RGB images in the images directory"""
        images = []
        for filename in os.listdir(os.path.join(APP_DIR, "images")):
            if filename.startswith("xs_original_rgb_") and filename.endswith(".png"):
                images.append(os.path.join("images/", filename))
        logger.info(f"Found {len(images)} available images")
        return images

    def get_random_image_selection(self, num_images=5):
        """Return a random selection of image filenames"""
        selected = random.sample(self.available_images, min(num_images, len(self.available_images)))
        return selected

    def reset(self, image_path):
        """Reset the model with a new starting image"""
        logger.info(f"Loading image: {image_path}")
        xs_rgb = (decode_image(image_path).float() / 255).unsqueeze(0).unsqueeze(0).to(self.device)
        logger.debug("Processing image...")
        self.xs = self.algo._maybe_encode(xs_rgb)
        self.conditions = torch.zeros(1, 1, self.algo.external_cond_dim, device=self.device)
        self.curr_frame = 1

    def encode_frame_to_base64(self, frame_data):
        """Convert a tensor frame to base64 encoded image string"""
        pil_img = Image.fromarray((frame_data[0, 0].clamp_(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy())
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @torch.no_grad()
    def generate_chunk(self, key):
        """Generate a chunk of frames based on a key press"""
        action_chunk = torch.zeros(self.algo.chunk_size, 1, self.algo.external_cond_dim, device=self.device)
        self.conditions = torch.cat([self.conditions, action_chunk], dim=0)
        assert self.conditions.shape[0] == self.curr_frame + self.algo.chunk_size

        scale = 0.5 if "Arrow" in key else 1.0
        if key == "ArrowUp":
            index = 2
            sign = +1
        elif key == "ArrowDown":
            index = 2
            sign = -1
        elif key == "ArrowLeft":
            index = 1
            sign = +1
        elif key == "ArrowRight":
            index = 1
            sign = -1
        elif key == "i":
            index = 0
            sign = +1
        elif key == "o":
            index = 0
            sign = -1
        elif key == "[":
            index = 6
            scale = 1
            sign = +1
        elif key == "]":
            index = 6
            scale = 1
            sign = -1
        else: # Do nothing
            index = 0
            scale = 0
            sign = +1
        self.conditions[self.curr_frame : self.curr_frame + self.algo.chunk_size, :, index] = sign * scale

        scheduling_matrix = self.algo._generate_scheduling_matrix(self.algo.chunk_size)
        chunk = torch.randn(
            (self.algo.chunk_size, 1, *self.algo.x_stacked_shape), device=self.device
        )
        chunk = torch.clamp(chunk, -self.algo.clip_noise, self.algo.clip_noise)
        self.xs = torch.cat([self.xs, chunk], 0)

        # sliding window: only input the last n_tokens frames
        start_frame = max(0, self.curr_frame + self.algo.chunk_size - self.algo.n_tokens)

        for m in tqdm(range(scheduling_matrix.shape[0] - 1), desc="denoising loop"):
            from_noise_levels = np.concatenate(
                (np.zeros((self.curr_frame,), dtype=np.int64), scheduling_matrix[m])
            )[:, None].repeat(1, axis=1)
            to_noise_levels = np.concatenate(
                (
                    np.zeros((self.curr_frame,), dtype=np.int64),
                    scheduling_matrix[m + 1],
                )
            )[:, None].repeat(1, axis=1)

            from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

            self.xs[start_frame:] = self.algo.diffusion_model.sample_step(
                self.xs[start_frame:],
                self.conditions[start_frame : self.curr_frame + self.algo.chunk_size],
                from_noise_levels[start_frame:],
                to_noise_levels[start_frame:],
            )
            
            latest_clean_idx = (to_noise_levels == 0).nonzero()[-1][0]
            if latest_clean_idx >= self.curr_frame:
                xs = self.algo._unstack_and_unnormalize(self.xs[latest_clean_idx:latest_clean_idx+1])
                xs_rgb = self.algo._maybe_decode(xs)
                yield latest_clean_idx, xs_rgb
        self.curr_frame += self.algo.chunk_size

    def get_frame(self, idx):
        """Get a frame at the specified index"""
        xs = self.algo._unstack_and_unnormalize(self.xs[idx:idx+1])
        xs_rgb = self.algo._maybe_decode(xs)
        return xs_rgb

    def debug(self):
        """Run a simple test of the world model."""
        rgb_list = []
        for chunk_idx in range(3):
            for i, xs_rgb in self.generate_chunk("d"):
                rgb_list.append((xs_rgb[0,0].permute(1, 2, 0).cpu() * 255).clamp_(0, 255).byte())
        rgb_list = torch.stack(rgb_list)
        write_video("video.mp4", rgb_list, fps=10)
    
    async def send_initial_frame(self, user_id, websocket):
        """Send initial frame to a user"""
        # Get random image selection for new player
        random_images = self.get_random_image_selection()
        
        # Reset game for the new user
        self.reset(RGB_PATH)
        
        # Send initial frame
        pil_img = Image.fromarray((self.get_frame(0)[0, 0].clamp_(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy())
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'info',
            'message': 'Your turn! Start playing by using the arrow keys.',
            'user_id': user_id
        }))
        
        # Send initial frame
        await websocket.send(json.dumps({
            'type': 'frame', 
            'frame': img_str,
            'user_id': user_id
        }))
        
        # Send random image selection
        await websocket.send(json.dumps({
            'type': 'image_selection_options',
            'images': random_images,
            'user_id': user_id
        }))

    async def handle_keypress(self, user_id, key, websocket):
        """Handle keypress from a user"""
        logger.info(f"Received keypress from user {user_id}: {key}")
        
        for i, xs_rgb in self.generate_chunk(key):
            pil_img = Image.fromarray((xs_rgb[0, 0].clamp_(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy())
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            img_json = json.dumps({
                'type': 'frame', 
                'frame': img_str,
                'user_id': user_id
            })
            logger.info(f"Sending frame {i} to user {user_id}")
            await websocket.send(img_json)

    async def handle_image_selection(self, user_id, path, websocket):
        """Handle image selection from a user"""
        # Adjust the path to point to the correct location
        image_path = os.path.join(APP_DIR, path)
        logger.info(f"Received image selection from user {user_id}: {image_path}")
        
        # Reset the game with the selected image
        self.reset(image_path)
        
        # Get a new random selection of images
        random_images = self.get_random_image_selection()
        
        # Send the new random image selection
        await websocket.send(json.dumps({
            'type': 'image_selection_options',
            'images': random_images,
            'user_id': user_id
        }))
        
        # Send the first frame of the reset game
        pil_img = Image.fromarray((self.get_frame(0)[0, 0].clamp_(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy())
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        await websocket.send(json.dumps({
            'type': 'frame', 
            'frame': img_str,
            'user_id': user_id
        }))

    async def connect_to_coordinator(self, coordinator_url, backend_id):
        """Handle connection to the coordinator server and all user interactions."""
        while True:
            try:
                async with websockets.connect(coordinator_url) as websocket:
                    logger.info(f"Connected to coordinator at {coordinator_url}")
                    
                    # Register with coordinator
                    await websocket.send(json.dumps({
                        'type': 'backend_register',
                        'name': backend_id
                    }))
                    
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data['type'] == 'registration_successful':
                        logger.info(f"Registration successful, server ID: {data['server_id']}")
                    else:
                        logger.error(f"Registration failed: {data}")
                        await asyncio.sleep(5)
                        continue
                    
                    # Start heartbeat sender task
                    heartbeat_sender = asyncio.create_task(heartbeat_loop(websocket))
                    
                    # Main message handling loop
                    try:
                        while True:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            # User assignment message
                            if data['type'] == 'user_assigned':
                                user_id = data['user_id']
                                logger.info(f"New user assigned: {user_id}")
                                
                                # Send initial frame
                                await self.send_initial_frame(user_id, websocket)
                                
                            # Handle keypresses and other messages from users
                            elif 'user_id' in data:
                                user_id = data['user_id']
                                
                                if data['type'] == 'keypress':
                                    await self.handle_keypress(user_id, data['key'], websocket)
                                elif data['type'] == 'image_selection':
                                    await self.handle_image_selection(user_id, data['path'], websocket)
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from coordinator: {e}")
                    except Exception as e:
                        logger.error(f"Error handling coordinator message: {e}")
                        # Cancel heartbeat task before reconnecting
                        heartbeat_sender.cancel()
                        raise  # Re-raise to trigger reconnection
                    
            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
                logger.error(f"Coordinator connection error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
            except Exception as e:
                logger.exception(f"Unexpected error in coordinator connection: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting

async def heartbeat_loop(websocket):
    """Background task to send heartbeats to the coordinator"""
    while True:
        try:
            await websocket.send(json.dumps({
                'type': 'heartbeat',
                'is_available': True  # Always available to receive users
            }))
            await asyncio.sleep(5)  # Send heartbeat every 5 seconds
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
            logger.error(f"Error sending heartbeat - connection closed: {e}")
            break
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            break

async def main(coordinator_url, backend_id=None):
    """Run the backend server with the specified configuration."""
    # Initialize the world model
    logger.info("Loading model...")
    world_model = WorldModel()
    logger.info("Model loaded successfully")

    # Configure defaults if not provided
    if not backend_id:
        backend_id = f"backend-{uuid.uuid4().hex[:8]}"
        
    logger.info(f"Connecting to coordinator at {coordinator_url}")
    logger.info(f"Backend ID: {backend_id}")
    
    # Connect to the coordinator
    await world_model.connect_to_coordinator(coordinator_url, backend_id)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the backend server')
    parser.add_argument('--coordinator', type=str, required=True, 
                        help='URL of the coordinator server (e.g., ws://localhost:8701)')
    parser.add_argument('--id', type=str, default=None,
                        help='ID of this backend server (e.g., gpu1)')
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(
            coordinator_url=args.coordinator,
            backend_id=args.id
        ))
    except KeyboardInterrupt:
        logger.info("Backend shutting down")

