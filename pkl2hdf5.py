import os
import pickle
import h5py
import numpy as np
from tqdm import tqdm
import argparse

def pkl_to_hdf5(input_dir, output_file):
    """
    指定されたディレクトリのPKLファイルをHDF5形式に変換して保存します。

    Args:
        input_dir (str): PKLファイルが格納されているディレクトリのパス
        output_file (str): 保存先HDF5ファイルのパス
    """
    # 初期化
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
        '/observations/images/camera': [],
    }
    
    # ディレクトリ内のPKLファイルを取得
    pkl_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pkl')])
    if not pkl_files:
        print(f"指定されたディレクトリにPKLファイルが存在しません: {input_dir}")
        return False  # エラーが発生した場合はFalseを返す
    
    # データ読み込みと保存準備
    for pkl_file in tqdm(pkl_files, desc=f"Processing {input_dir}"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # 必要なデータを抽出
        position = np.array(data['position'], dtype=np.float32)  # Float型
        direction = np.array(data['direction'], dtype=np.float32)
        action = np.concatenate([position, direction], axis=-1)  # axis=-1 は最後の次元で結合
        velocity = np.array(data['velocity'], dtype=np.float32)  # Float型
        image = np.array(data['base_rgb'], dtype=np.uint8) 
        
        # データを追加
        data_dict['/observations/qpos'].append(position)
        data_dict['/observations/qvel'].append(velocity)
        data_dict['/action'].append(action)  # actionとしてpositionを使用
        data_dict['/observations/images/camera'].append(image)

    # リストをnumpy配列に変換
    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key])

    # HDF5ファイルに保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as hdf5_file:
        
        hdf5_file.attrs['sim'] = True
        # 観測データグループ
        obs_group = hdf5_file.create_group("observations")
        obs_group.create_dataset("qpos", data=data_dict['/observations/qpos'])
        obs_group.create_dataset("qvel", data=data_dict['/observations/qvel'])
        
        # カメラ画像データ
        images_group = obs_group.create_group("images")
        images_group.create_dataset(
            "camera", 
            data=data_dict['/observations/images/camera'], 
            dtype='uint8', 
            chunks=(1, *data_dict['/observations/images/camera'].shape[1:])
        )
        
        # アクションデータ
        hdf5_file.create_dataset("action", data=data_dict['/action'], dtype='float32')

    print(f"HDF5ファイルが保存されました: {output_file}")
    return True  # 正常終了

if __name__ == "__main__":
    # 実行時引数のパース
    parser = argparse.ArgumentParser(description="PKLからHDF5への変換スクリプト")
    parser.add_argument("--begin", type=int, required=True, help="シーンIDの開始 (例: 0)")
    parser.add_argument("--end", type=int, required=True, help="シーンIDの終了 (例: 10)")
    args = parser.parse_args()
    
    # 指定された範囲のシーンを変換
    for scene_id in range(args.begin, args.end + 1):
        input_dir = f"/home/projects/ACT/dataset/output/data/scene_{scene_id:05d}/run_00"
        output_file = f"/home/projects/ACT/dataset/output/hdf5/task1/episode_{scene_id}.hdf5"
        
        # 変換処理
        if not pkl_to_hdf5(input_dir, output_file):
            print(f"シーン{scene_id}の変換に失敗しました。")
