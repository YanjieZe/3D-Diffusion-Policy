import zarr

# Open a Zarr array from a directory
# 'my_zarr_array.zarr' represents the directory containing the Zarr array
zarr_array = zarr.open('3D-Diffusion-Policy/data/realdex_drill.zarr/data/action', mode='r')
import pdb; pdb.set_trace()
# Access data
print(zarr_array[:]) # Read the entire array into memory
print(zarr_array[0, 0]) # Access a specific element
print(zarr_array[0:10, :]) # Access a slice of the array
