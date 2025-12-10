import h5py
import numpy as np

def print_hdf5_structure(file_path):
    """Recursively prints the structure of an HDF5 file."""
    print(f"--- Structure of: {file_path} ---")

    try:
        with h5py.File(file_path, 'r') as f:
            f.visititems(visitor_func)
    except Exception as e:
        print(f"Error opening file: {e}")

def visitor_func(name, obj):
    """Visitor function called for every item (Group or Dataset) in the file."""
    indent = ' ' * (name.count('/') * 4)

    if isinstance(obj, h5py.Group):
        print(f"{indent}GROUP: {name}/")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}DATASET: {name}")
        print(f"{indent}    - Shape: {obj.shape}")
        print(f"{indent}    - Dtype: {obj.dtype}")

        if obj.attrs:
            print(f"{indent}    - Attributes: {list(obj.attrs.items())[:2]}...")

if __name__ == "__main__":
    # --- ADD THIS LINE TO EXECUTE THE INSPECTION ---
    # Replace this with the path to one of your training data files, e.g., indoor_flying2_data.hdf5
    data_file_path = './data/indoor_flying4_data.hdf5'
    print_hdf5_structure(data_file_path)