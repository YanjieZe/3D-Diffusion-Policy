import os
import zarr
import numpy as np
import rerun as rr
import argparse
from tqdm import tqdm
import time
def visualize_trajectory_headless(zarr_path, episode_idx=0, output_path=None):
    """
    Visualize a trajectory from a Zarr file using Rerun Web Viewer for headless servers.
    """
    print(f"Opening Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    episode_ends = root['meta']['episode_ends'][:]
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    
    print(f"Extracting episode {episode_idx} with range [{start_idx}, {end_idx})")
    pc_data = root['data']['point_cloud'][start_idx:end_idx]
    
    has_colors = pc_data.shape[-1] == 6

    # Initialize rerun
    rr.init("Point Cloud Trajectory", spawn=False)
    
    # This hosts the viewer and the websocket server.
    # You MUST forward both ports for the web viewer to work.
    web_port = 9091
    ws_port = 9877
    
    print("\n" + "="*70)
    print("RERUN WEB SERVER STARTING")
    print(f"1. Open your terminal on your LOCAL machine and run:")
    print(f"   ssh -L {web_port}:localhost:{web_port} -L {ws_port}:localhost:{ws_port} user@your-server-ip")
    print(f"\n2. Then open this URL in your LOCAL browser:")
    print(f"   http://localhost:{web_port}")
    print("="*70 + "\n")
    
    # serve_web hosts both the WASM viewer and the websocket data stream
    rr.serve_web(
        web_port=web_port,   # Port for the web viewer (HTTP)
        ws_port=ws_port,     # Port for the data stream (WebSocket)
        open_browser=False
    )

    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"Saving recording to {output_path}...")
        rr.save(output_path)

    print("IMPORTANT: Open the link above BEFORE pressing Enter.")
    input("Press Enter to start streaming data to Rerun...")
    
    print("Streaming data...")
    for t in tqdm(range(len(pc_data))):
        rr.set_time_sequence("step", t)
        rr.set_time_seconds("time", t / 10.0)
        
        current_pc = pc_data[t]
        xyz = current_pc[:, :3]
        
        if has_colors:
            rgb = current_pc[:, 3:]
            colors = rgb if rgb.max() <= 1.001 else rgb / 255.0
            rr.log("world/point_cloud", rr.Points3D(positions=xyz, colors=colors, radii=0.003))
        else:
            z_norm = (xyz[:, 2] - xyz[:, 2].min()) / (xyz[:, 2].max() - xyz[:, 2].min() + 1e-6)
            colors = np.stack([z_norm, np.zeros_like(z_norm), 1.0 - z_norm], axis=-1)
            rr.log("world/point_cloud", rr.Points3D(positions=xyz, colors=colors, radii=0.003))
        
        # Tiny sleep to avoid overwhelming the socket during the initial burst
        time.sleep(0.005)

    print("\nData streaming complete. Keep this script running to keep the web server alive.")
    # Block so the server stays up
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, default="/scratch2/cross-emb/DP3_data/data_from_puru_no_eef_binary_gripper.zarr")
    parser.add_argument('--episode', type=int, default=4)
    parser.add_argument('--output', type=str, default=None, help="Save the Rerun recording to this .rrd file")
    args = parser.parse_args()
    
    visualize_trajectory_headless(args.zarr_path, args.episode, args.output)
