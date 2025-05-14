import asyncio
import json
import time
import random
import websockets
from collections import deque
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('coordinator')

# Default ports
DEFAULT_USER_PORT = 8700
DEFAULT_BACKEND_PORT = 8701

# Heartbeat timeouts
USER_HEARTBEAT_TIMEOUT = 5  # Seconds
BACKEND_HEARTBEAT_TIMEOUT = 10  # Seconds

# Session timeout
SESSION_TIMEOUT = 60  # Seconds

class CoordinatorServer:
    """Coordinator server that manages connections between users and backends."""
    
    def __init__(self):
        # Queue of waiting user IDs
        self.global_queue = deque()
        
        # Backend state
        self.backend2websocket = {}  # Map of backend_id -> websocket object
        self.backend2heartbeat = {}  # Map of backend_id -> last heartbeat time
        self.backend2user = {}       # Map of backend_id -> user_id assigned to it (or None if available)
        
        # User state
        self.user2websocket = {}     # Map of user_id -> websocket object
        self.user2heartbeat = {}     # Map of user_id -> last heartbeat time
        self.user2backend = {}       # Map of user_id -> backend_id assigned to it (or None if not assigned)
        self.user2sessionstart = {}  # Map of user_id -> timestamp when they were assigned to a backend
    
    async def send_to_user(self, user_id, message_data):
        """Send a message to a user."""
        if user_id not in self.user2websocket:
            logger.warning(f"Cannot send message: user {user_id} not connected")
            return False
        
        websocket = self.user2websocket[user_id]
        try:
            await websocket.send(json.dumps(message_data))
            return True
        except Exception as e:
            logger.error(f"Error sending message to user {user_id}: {e}")
            return False
    
    async def send_to_backend(self, backend_id, message_data):
        """Send a message to a backend."""
        if backend_id not in self.backend2websocket:
            logger.warning(f"Cannot send message: backend {backend_id} not connected")
            return False
        
        websocket = self.backend2websocket[backend_id]
        try:
            await websocket.send(json.dumps(message_data))
            return True
        except Exception as e:
            logger.error(f"Error sending message to backend {backend_id}: {e}")
            return False
    
    async def register_backend(self, backend_id, websocket):
        """Register a new backend server."""
        current_time = time.time()
        
        # Check if backend already exists
        if backend_id in self.backend2websocket:
            # Update existing backend
            self.backend2websocket[backend_id] = websocket
            self.backend2heartbeat[backend_id] = current_time
            logger.info(f"Updated existing backend: {backend_id}")
        else:
            # Create new backend entry
            self.backend2websocket[backend_id] = websocket
            self.backend2heartbeat[backend_id] = current_time
            self.backend2user[backend_id] = None  # Not assigned to any user initially
            logger.info(f"Registered new backend: {backend_id}")
        
        # Process queue in case users are waiting
        await self.process_queue()
        
        return backend_id
    
    async def find_available_backend(self):
        """Find an available backend server."""
        current_time = time.time()
        
        for backend_id in list(self.backend2websocket.keys()):
            # Check if backend is available and not assigned to a user
            if backend_id in self.backend2user and self.backend2user[backend_id] is None:
                # Check if backend is still alive
                if (current_time - self.backend2heartbeat[backend_id]) > BACKEND_HEARTBEAT_TIMEOUT:
                    # Backend timed out, remove it
                    logger.warning(f"Backend {backend_id} timed out, removing")
                    self.remove_backend(backend_id)
                    continue
                return backend_id
        
        return None
    
    def remove_backend(self, backend_id):
        """Remove a backend from all mappings."""
        # Get the user assigned to this backend, if any
        user_id = self.backend2user.get(backend_id)
        
        # Remove backend from all mappings
        self.backend2websocket.pop(backend_id, None)
        self.backend2heartbeat.pop(backend_id, None)
        self.backend2user.pop(backend_id, None)
        
        # Return the user ID if one was assigned
        return user_id
    
    async def assign_user_to_backend(self, user_id):
        """Assign a user to an available backend server."""
        # Check if user exists
        if user_id not in self.user2websocket:
            logger.warning(f"Cannot assign user {user_id} to backend: user not found")
            return False
        
        # Find an available backend
        backend_id = await self.find_available_backend()
        
        if backend_id is None:
            # No available backends, keep user in queue
            logger.info(f"No available backends for user {user_id}, keeping in queue")
            return False
        
        # Assign user to backend
        self.backend2user[backend_id] = user_id
        self.user2backend[user_id] = backend_id
        
        # Record session start time
        self.user2sessionstart[user_id] = time.time()
        
        logger.info(f"Assigned user {user_id} to backend {backend_id}")
        
        # Send assignment message to user
        await self.send_to_user(user_id, {
            'type': 'backend_assigned',
            'backend_name': backend_id,
            'session_timeout': SESSION_TIMEOUT
        })
        
        # Send initial message to backend about the new user
        success = await self.send_to_backend(backend_id, {
            'type': 'user_assigned',
            'user_id': user_id
        })
        
        if not success:
            # Handle failure by making backend available again
            self.backend2user[backend_id] = None
            self.user2backend.pop(user_id, None)
            self.user2sessionstart.pop(user_id, None)
            return False
        
        return True
    
    async def process_queue(self):
        """Process the global queue, assigning users to available backends."""
        if not self.global_queue:
            return
        
        # Process users in queue until no more can be assigned
        assigned_count = 0
        queue_copy = list(self.global_queue)  # Make a copy to avoid modifying during iteration
        
        for user_id in queue_copy:
            if user_id not in self.user2websocket:
                # User disconnected, remove from queue
                self.global_queue.remove(user_id)
                continue
            
            success = await self.assign_user_to_backend(user_id)
            
            if success:
                # Remove user from queue
                self.global_queue.remove(user_id)
                assigned_count += 1
                await self.update_queue_positions()
            else:
                # No more backends available
                break
        
        if assigned_count > 0:
            logger.info(f"Assigned {assigned_count} users from queue")
    
    async def update_queue_positions(self):
        """Update all users in queue with their current position."""
        for i, user_id in enumerate(self.global_queue):
            if user_id in self.user2websocket:
                await self.send_to_user(user_id, {
                    'type': 'queue_position',
                    'position': i + 1,
                    'total_in_queue': len(self.global_queue)
                })
    
    async def release_user_from_backend(self, user_id, backend_id):
        """Release a user from a backend server."""
        if backend_id in self.backend2user and self.backend2user[backend_id] == user_id:
            # Release user
            self.backend2user[backend_id] = None
            logger.info(f"Released user {user_id} from backend {backend_id}")
            
            # Update user state if user still exists
            self.user2backend.pop(user_id, None)
            self.user2sessionstart.pop(user_id, None)
            
            # Notify the backend that the user has been released
            await self.send_to_backend(backend_id, {
                'type': 'user_released',
                'user_id': user_id
            })
            
            # Notify the user that their session has ended
            if user_id in self.user2websocket:
                await self.send_to_user(user_id, {
                    'type': 'info',
                    'message': 'Your session has ended. You have been placed back in the queue.'
                })
                
                # Add the user back to the queue
                self.global_queue.append(user_id)
                await self.update_queue_positions()
            
            # Process queue to assign new user if available
            await self.process_queue()
            return True
        
        return False
    
    async def relay_to_backend(self, user_id, message_data):
        """Relay a message from a user to its assigned backend server."""
        if user_id not in self.user2backend or self.user2backend[user_id] is None:
            logger.warning(f"Cannot relay message: user {user_id} not assigned to any backend")
            return False
        
        backend_id = self.user2backend[user_id]
        
        # Add user_id to the message so backend knows which user it's from
        message_data['user_id'] = user_id
        return await self.send_to_backend(backend_id, message_data)
    
    async def relay_to_user(self, user_id, message_data):
        """Relay a message from a backend to a specific user."""
        if user_id not in self.user2websocket:
            logger.warning(f"Cannot relay message: user {user_id} not connected")
            return False
        
        # Remove user_id from the message if it exists
        if 'user_id' in message_data:
            del message_data['user_id']
        
        return await self.send_to_user(user_id, message_data)
    
    async def handle_backend(self, websocket, path=None):
        """Handle full lifecycle of a backend server connection."""
        backend_id = None
        
        try:
            # Register the backend
            msg = await websocket.recv()
            data = json.loads(msg)
            
            if data['type'] != 'backend_register':
                logger.error(f"Unexpected message type from backend: {data['type']}")
                return
            
            backend_id = data['name']
            await self.register_backend(backend_id, websocket)
            
            # Send confirmation
            await websocket.send(json.dumps({
                'type': 'registration_successful',
                'server_id': backend_id
            }))
            
            # Handle ongoing communication
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'heartbeat':
                    # Update last heartbeat time
                    self.backend2heartbeat[backend_id] = time.time()
                    
                    # Update server status if provided
                    if 'is_available' in data and not data['is_available']:
                        # If backend reports as unavailable, don't assign new users to it
                        # but keep existing connections
                        if self.backend2user[backend_id] is None:
                            logger.info(f"Backend {backend_id} marked as unavailable")
                    
                    # Handle user release if needed
                    if data.get('user_released') and 'user_id' in data:
                        await self.release_user_from_backend(data['user_id'], backend_id)
                
                elif data['type'] == 'status_update':
                    # Update server status
                    if 'is_available' in data and data['is_available'] and self.backend2user[backend_id] is None:
                        # Backend became available again, process queue
                        logger.info(f"Status update from {backend_id}: available=True")
                        await self.process_queue()
                
                # Relay messages from backend to user
                elif 'user_id' in data and data.get('type') != 'user_released':
                    user_id = data['user_id']
                    await self.relay_to_user(user_id, data)
        
        except websockets.exceptions.ConnectionClosed:
            if backend_id:
                logger.warning(f"Backend connection closed: {backend_id}")
        except Exception as e:
            logger.exception(f"Error in backend connection: {e}")
        finally:
            # Clean up resources if backend_id was set
            if backend_id and backend_id in self.backend2websocket:
                user_id = self.remove_backend(backend_id)
                
                # If a user was assigned to this backend, handle it
                if user_id and user_id in self.user2websocket:
                    # Notify user about backend disconnect
                    await self.send_to_user(user_id, {
                        'type': 'info',
                        'message': 'Backend server disconnected. You have been placed back in the queue.'
                    })
                    
                    # Clear user's backend assignment
                    self.user2backend.pop(user_id, None)
                    self.user2sessionstart.pop(user_id, None)
                    
                    # Add to queue
                    self.global_queue.append(user_id)
                    logger.info(f"Added user {user_id} back to queue after backend disconnect")
                    
                    # Update queue positions
                    await self.update_queue_positions()
    
    async def check_dead_connections(self):
        """Periodically check for dead connections and session timeouts."""
        while True:
            current_time = time.time()
            
            # Check for session timeouts
            timed_out_sessions = []
            for user_id, session_start in self.user2sessionstart.items():
                if (current_time - session_start) > SESSION_TIMEOUT:
                    logger.info(f"User {user_id} session timed out after {SESSION_TIMEOUT} seconds")
                    timed_out_sessions.append(user_id)
            
            # Handle session timeouts
            for user_id in timed_out_sessions:
                if user_id in self.user2backend and self.user2backend[user_id] is not None:
                    backend_id = self.user2backend[user_id]
                    await self.release_user_from_backend(user_id, backend_id)
            
            # Check for dead backends
            dead_backends = []
            for backend_id, last_heartbeat in self.backend2heartbeat.items():
                if (current_time - last_heartbeat) > BACKEND_HEARTBEAT_TIMEOUT:
                    logger.warning(f"Backend {backend_id} timed out, removing")
                    dead_backends.append(backend_id)
            
            # Handle users of dead backends
            for backend_id in dead_backends:
                user_id = self.remove_backend(backend_id)
                
                # If a user was assigned to this backend, handle it
                if user_id and user_id in self.user2websocket:
                    # Notify user
                    await self.send_to_user(user_id, {
                        'type': 'info',
                        'message': 'Backend server disconnected. You have been placed back in the queue.'
                    })
                    
                    # Clear user's backend assignment
                    self.user2backend.pop(user_id, None)
                    self.user2sessionstart.pop(user_id, None)
                    
                    # Add to queue
                    self.global_queue.append(user_id)
                    
                    # Update queue positions
                    await self.update_queue_positions()
            
            # Check for dead users
            dead_users = []
            for user_id, last_heartbeat in self.user2heartbeat.items():
                if (current_time - last_heartbeat) > USER_HEARTBEAT_TIMEOUT:
                    logger.warning(f"User {user_id} timed out (no heartbeat)")
                    dead_users.append(user_id)
            
            # Handle dead users
            for user_id in dead_users:
                # Remove from user mappings
                self.user2websocket.pop(user_id, None)
                self.user2heartbeat.pop(user_id, None)
                self.user2sessionstart.pop(user_id, None)
                
                # Remove from queue if present
                if user_id in self.global_queue:
                    self.global_queue.remove(user_id)
                    await self.update_queue_positions()
                
                # Release from backend if assigned
                if user_id in self.user2backend and self.user2backend[user_id] is not None:
                    backend_id = self.user2backend[user_id]
                    await self.release_user_from_backend(user_id, backend_id)
                
                # Clean up user-to-backend mapping
                self.user2backend.pop(user_id, None)
            
            # Process queue to assign users to available backends
            await self.process_queue()
            
            await asyncio.sleep(2)  # Check every 2 seconds
    
    async def handle_user(self, websocket, path=None):
        """Handle user WebSocket connection."""
        # Generate user ID
        user_id = f"user_{time.time()}_{random.randint(1000, 9999)}"
        
        # Create user entry
        self.user2websocket[user_id] = websocket
        self.user2heartbeat[user_id] = time.time()
        self.user2backend[user_id] = None  # Not assigned to any backend initially
        
        # Initialize user connection
        await websocket.send(json.dumps({
            'type': 'info',
            'message': 'Connected to coordinator server'
        }))
        
        # Send user ID
        await websocket.send(json.dumps({
            'type': 'user_id',
            'id': user_id
        }))
        
        logger.info(f"New user connected: {user_id}")
        
        try:
            # Try to assign user to an available backend immediately
            if await self.assign_user_to_backend(user_id):
                # Successfully assigned
                logger.info(f"Immediately assigned user {user_id} to a backend")
            else:
                # No available backends, add user to queue
                self.global_queue.append(user_id)
                logger.info(f"Added user {user_id} to queue (position {len(self.global_queue)})")
                
                # Send queue position
                await websocket.send(json.dumps({
                    'type': 'queue_position',
                    'position': len(self.global_queue),
                    'total_in_queue': len(self.global_queue)
                }))
                
                # Update all users' queue positions
                await self.update_queue_positions()
            
            # Handle ongoing communication
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Update heartbeat time
                    if data['type'] == 'heartbeat':
                        self.user2heartbeat[user_id] = time.time()
                        continue
                    
                    # Relay all messages to the assigned backend
                    if user_id in self.user2backend and self.user2backend[user_id] is not None:
                        await self.relay_to_backend(user_id, data)
                    else:
                        logger.warning(f"Received message from user {user_id} not assigned to a backend")
                        # Inform user they're not assigned to a backend
                        await websocket.send(json.dumps({
                            'type': 'info',
                            'message': 'You are not currently assigned to a backend. Please wait in the queue.'
                        }))
                
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from user {user_id}: {message}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"User disconnected: {user_id}")
        except Exception as e:
            logger.exception(f"Error handling user {user_id}: {e}")
        finally:
            # Clean up user resources
            self.user2websocket.pop(user_id, None)
            self.user2heartbeat.pop(user_id, None)
            
            # Remove from queue if present
            if user_id in self.global_queue:
                self.global_queue.remove(user_id)
                await self.update_queue_positions()
            
            # Release from backend if assigned
            if user_id in self.user2backend and self.user2backend[user_id] is not None:
                backend_id = self.user2backend[user_id]
                await self.release_user_from_backend(user_id, backend_id)
            
            # Clean up user-to-backend mapping
            self.user2backend.pop(user_id, None)
    
    async def start(self, host, user_port, backend_port):
        """Start the coordinator server."""
        # Create handler functions for websockets that call our instance methods
        async def user_handler(websocket, path=None):
            await self.handle_user(websocket, path)
            
        async def backend_handler(websocket, path=None):
            await self.handle_backend(websocket, path)
        
        # Create different handlers for users and backends
        user_server = websockets.serve(user_handler, host, user_port)
        backend_server = websockets.serve(backend_handler, host, backend_port)
        
        # Start dead connection checker
        connection_checker = asyncio.create_task(self.check_dead_connections())
        
        logger.info("Coordinator starting up")
        logger.info(f"User server listening on ws://{host}:{user_port}")
        logger.info(f"Backend server listening on ws://{host}:{backend_port}")
        
        # Start all servers
        await asyncio.gather(
            user_server,
            backend_server,
            connection_checker
        )

async def main(host, user_port, backend_port):
    """Main entry point for running the coordinator server."""
    coordinator = CoordinatorServer()
    await coordinator.start(host, user_port, backend_port)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the coordinator server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to listen for connections (default: 0.0.0.0)')
    parser.add_argument('--user-port', type=int, default=DEFAULT_USER_PORT,
                        help=f'Port to listen for user connections (default: {DEFAULT_USER_PORT})')
    parser.add_argument('--backend-port', type=int, default=DEFAULT_BACKEND_PORT,
                        help=f'Port to listen for backend connections (default: {DEFAULT_BACKEND_PORT})')
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.host, args.user_port, args.backend_port))
    except KeyboardInterrupt:
        logger.info("Coordinator shutting down") 
