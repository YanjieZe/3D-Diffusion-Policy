import os
import sys
import pathlib
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Any, List
import argparse
import base64
import cv2
import time


# Setup python path to match train.py environment
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

# Import TrainDP3Workspace to load the model
try:
    from train import TrainDP3Workspace
except ImportError:
    # If running from a different directory, try appending the current directory
    sys.path.append(os.getcwd())
    from train import TrainDP3Workspace

from diffusion_policy_3d.common.pytorch_util import dict_apply
from get_filtered_pc import rgbd_to_sampled_pc, DEFAULT_INTRINSICS


app = FastAPI()
DEVICE = "cuda:0"
WORKSPACE = None

class InferenceRequest(BaseModel):
    # Expect a dictionary containing "point_cloud" and "agent_pos"
    # values should be lists representing the time sequence
    observation: Dict[str, Any]

@app.on_event("startup")
def load_model():
    global WORKSPACE
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        print("Warning: CHECKPOINT_PATH env var not set. Pass --checkpoint argument or set env var.")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        # Load the workspace from checkpoint
        # This restores the model, configuration, and normalizer
        WORKSPACE = TrainDP3Workspace.create_from_checkpoint(checkpoint_path)
        WORKSPACE.model.eval()
        WORKSPACE.model.to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We don't exit here to allow debugging or delayed loading
        
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global WORKSPACE
    await websocket.accept()
    
    if WORKSPACE is None:
        await websocket.close(code=1011, reason="Model not initialized.")
        return

    try:
        while True:
            # Receive JSON payload
            data = await websocket.receive_json()
            request = InferenceRequest(observation=data['observation'])

            obs_dict_input = request.observation
            t_start = time.time()
            t_img_start = time.time()
            # Handle RGB/Depth to Point Cloud Conversion
            if 'rgb' in obs_dict_input and 'depth' in obs_dict_input and 'point_cloud' not in obs_dict_input:
                # Validate input format
                # Expecting list of base64 strings for T steps
                rgb_list = obs_dict_input['rgb']
                depth_list = obs_dict_input['depth']
                
                if len(rgb_list) != len(depth_list):
                     # Cannot raise HTTP exception in websocket, just close or ignore
                     print("Mismatch in RGB and Depth sequence length.")
                     continue
                     
                pc_sequence = []
                
                for i in range(len(rgb_list)):
                    # Decode RGB
                    # Assumes Base64 encoded PNG or JPG or similar that cv2.imdecode can handle
                    rgb_bytes = base64.b64decode(rgb_list[i])
                    rgb_arr = np.frombuffer(rgb_bytes, dtype=np.uint8)
                    rgb_img = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # Ensure RGB
                    
                    # Decode Depth
                    # Assumes Base64 encoded 16-bit PNG (standard for depth) or similar
                    # If raw bytes, we need shape. 
                    # Let's try cv2 decode first.
                    depth_bytes = base64.b64decode(depth_list[i])
                    depth_arr = np.frombuffer(depth_bytes, dtype=np.uint8)
                    depth_img = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
                    
                    # Convert Depth to Meters
                    # Realsense usually matches 1000 units = 1 meter for uint16
                    if depth_img.dtype == np.uint16:
                         depth_img = depth_img.astype(np.float32) / 1000.0
                    
                    # Generate Point Cloud
                    # Using defaults for intrinsics for now, or could pass in request
                    xyz, colors = rgbd_to_sampled_pc(rgb_img, depth_img, num_points=2500) # NOTE: verify num_points vs model
                    
                    # Stack XYZ and RGB -> (N, 6)
                    # Ensure colors are float 0-1 (rgbd_to_sampled_pc does this)
                    pc = np.concatenate([xyz, colors], axis=-1)
                    pc_sequence.append(pc)
                    
                # Update obs_dict_input with point_cloud
                obs_dict_input['point_cloud'] = pc_sequence
                
                # Remove raw images to save memory/avoid issues if model doesn't expect them
                del obs_dict_input['rgb']
                del obs_dict_input['depth']

            
            t_img_end = time.time()

            def to_tensor(x):
                # Convert list to numpy if needed
                if isinstance(x, list):
                    x = np.array(x)
                # Create tensor
                # Ensure float32 for floating point data
                if x.dtype == np.float64:
                     x = x.astype(np.float32)

                t = torch.from_numpy(x)
                
                # Move to device
                t = t.to(DEVICE)
                
                # Add batch dimension: (T, ...) -> (1, T, ...)
                t = t.unsqueeze(0)
                return t

            obs_dict = dict_apply(obs_dict_input, to_tensor)

            t_infer_start = time.time()
            with torch.no_grad():
                # policy.predict_action handles normalization internally using the loaded normalizer
                result = WORKSPACE.model.predict_action(obs_dict)
                action = result['action_pred'][:,1:]
            t_infer_end = time.time()
            
            print(f"Latency | Total: {(t_infer_end - t_start)*1000:.1f}ms | Image Proc: {(t_img_end - t_img_start)*1000:.1f}ms | Inference: {(t_infer_end - t_infer_start)*1000:.1f}ms")

            # Action shape: (B, T, D) -> (1, T, D)
            # Return as list, removing batch dimension
            action_list = action[0].cpu().numpy().tolist()
            
            await websocket.send_json({"action": action_list})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file")
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    if args.checkpoint:
        os.environ["CHECKPOINT_PATH"] = args.checkpoint
        
    uvicorn.run(app, host=args.host, port=args.port)
