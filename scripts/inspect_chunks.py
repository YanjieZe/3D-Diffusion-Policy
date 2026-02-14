import zarr
import os

def inspect_zarr_chunks(path):
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("Error: File not found!")
        return

    try:
        z = zarr.open(path, 'r')
        
        print("\n=== Hierarchy ===")
        print(z.tree())
        
        print("\n=== Data Array Chunks ===")
        if 'data' in z:
            for key, val in z['data'].items():
                print(f"Array: data/{key}")
                print(f"  Shape:  {val.shape}")
                print(f"  Chunks: {val.chunks}")
                print(f"  Type:   {val.dtype}")
                # Calculate chunk size in MB for context
                chunk_bytes = 1
                for s, c in zip(val.shape, val.chunks):
                    pass # logic for specific chunk size
                    
        print("\n=== Meta Array Chunks ===")
        if 'meta' in z:
            for key, val in z['meta'].items():
                print(f"Array: meta/{key}")
                print(f"  Shape:  {val.shape}")
                print(f"  Chunks: {val.chunks}")

    except Exception as e:
        print(f"Error reading zarr: {e}")

if __name__ == "__main__":
    inspect_zarr_chunks('/home/cross-emb/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_robot_demo/drill_40demo_1024.zarr')
