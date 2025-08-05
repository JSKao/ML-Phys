from sklearn.model_selection import train_test_split

def load_data(
    filename,
    N=None,
    y_cols=2,
    test_size=0.2,
    random_state=42,
    x_key="x",
    y_key="y",
    h5_x_path=None,
    h5_y_path=None,
    graph_key=None
):
    """
    萬用資料載入器，支援 npz/csv/npy/h5/graph
    - filename: 檔名
    - N: 輸入維度（csv/npy 必填，npz/h5 可自動判斷）
    - y_cols: label 欄位數（csv/npy 用）
    - x_key, y_key: npz/h5 檔案內的 key
    - h5_x_path, h5_y_path: h5 檔案內的完整路徑（如 group/dataset）
    - graph_key: 若 npz/h5 內含圖結構資料，請指定 key
    - 回傳: x_train, x_test, y_train, y_test, N 或特殊資料型態
    """
    # --- 圖結構資料（假設以 npz/h5 儲存 adjacency matrix 或 edge list）---
    if graph_key is not None:
        if filename.endswith('.npz'):
            npz = np.load(filename, allow_pickle=True)
            graph_data = npz[graph_key]
            print("Loaded graph data from npz file.")
            return graph_data
        elif filename.endswith('.h5') or filename.endswith('.hdf5'):
            import h5py
            with h5py.File(filename, "r") as f:
                graph_data = f[graph_key][()]
            print("Loaded graph data from h5 file.")
            return graph_data
        else:
            raise ValueError("圖結構資料目前只支援 npz/h5 格式")
    
    # --- 一般資料 ---
    if filename.endswith('.npz'):
        npz = np.load(filename)
        x_data = npz[x_key]
        y_data = npz[y_key]
        N = x_data.shape[1] if x_data.ndim == 2 else np.prod(x_data.shape[1:])
        print(f"Loaded from npz file. Data shape: {x_data.shape}")
    elif filename.endswith('.csv'):
        if N is None:
            raise ValueError("請指定 N（csv 格式必須指定輸入維度）")
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        x_data = data[:, :N]
        y_data = data[:, N:N+y_cols]
        print(f"Loaded from CSV file. Data shape: {x_data.shape}")
    elif filename.endswith('.npy'):
        arr = np.load(filename)
        if arr.ndim == 2:
            if N is None:
                raise ValueError("請指定 N（npy 格式必須指定輸入維度）")
            x_data = arr[:, :N]
            y_data = arr[:, N:N+y_cols]
        else:
            # 影像/張量/時序等
            x_data = arr
            y_data = None
        N = x_data.shape[1] if x_data.ndim == 2 else np.prod(x_data.shape[1:])
        print(f"Loaded from npy file. Data shape: {x_data.shape}")
    elif filename.endswith('.h5') or filename.endswith('.hdf5'):
        import h5py
        with h5py.File(filename, "r") as f:
            if h5_x_path is None or h5_y_path is None:
                raise ValueError("請指定 h5_x_path, h5_y_path")
            x_data = f[h5_x_path][()]
            y_data = f[h5_y_path][()]
            N = x_data.shape[1] if x_data.ndim == 2 else np.prod(x_data.shape[1:])
        print(f"Loaded from h5 file. Data shape: {x_data.shape}")
    else:
        raise ValueError("只支援 .npz, .csv, .npy, .h5 檔案")
    
    # --- 自動辨識資料型態並處理 ---
    if y_data is not None and y_data.ndim == 2:
        # 表格/向量型資料
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_size, random_state=random_state
        )
        # 轉置 (N, m)
        x_train = x_train.T
        x_test = x_test.T
        y_train = y_train.T
        y_test = y_test.T
        print("表格/向量型資料，已切分訓練/測試集並轉置")
        print("x_train, y_train, x_test, y_test shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        return x_train, x_test, y_train, y_test, N
    elif y_data is None and x_data.ndim in [2, 3, 4]:
        # 影像/張量/時序型資料
        print(f"影像/張量/時序型資料，shape: {x_data.shape}")
        return x_data, N
    else:
        print("資料型態未自動辨識，請手動檢查")
        return x_data, y_data, N