import h5py

file_path = "/home/projects/ACT/data/output/hdf5/task1/episode_0.hdf5"
with h5py.File(file_path, "r") as f:
    print("Keys in HDF5 file:", list(f.keys()))  # トップレベルのグループを表示
    print("Shape of qpos:", f["observations/qpos"].shape)
    print("Shape of qvel:", f["observations/qvel"].shape)
    print("Shape of action:", f["action"].shape)