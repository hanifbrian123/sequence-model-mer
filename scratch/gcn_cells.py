---
## Step 3: Dataset 7-Channel Feature Extraction & Precomputed Cache
# --- CELL ---
# ============================================================================
# 1. TOPOLOGI MATRIKS ADJASENSI WAJAH (MEDAPIPE FACEMESH)
# ============================================================================
def build_normalized_adjacency(num_nodes=468):
    """
    Membangun matriks adiasensi spasial simetris ternormalisasi D^(-1/2) * A * D^(-1/2).
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    # Self-loops
    A = A + np.eye(num_nodes, dtype=np.float32)
    
    # Symmetric Normalization
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=D > 0)
    D_inv_sqrt[D == 0] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat
    
    return torch.tensor(A_norm, dtype=torch.float32)

A_TENSOR = build_normalized_adjacency(num_nodes=NUM_NODES).to(DEVICE)
print(f"✅ Matriks Adiasensi Wajah Siap! Dimensi: {A_TENSOR.shape}")
# --- CELL ---
# ============================================================================
# 2. PRECOMPUTED GRAPH DATASET READER (HIGH-SPEED RAM CACHE)
# ============================================================================
PRECOMPUTED_DIR = os.path.join(PROJECT_ROOT, "precomputed_graphs_7ch_real_CASME2")

class PrecomputedGraphDataset(torch.utils.data.Dataset):
    """
    Dataset graf 7-channel super cepat yang membaca file npy hasil pra-komputasi.
    """
    def __init__(self, precomputed_dir, num_classes=5):
        self.precomputed_dir = precomputed_dir
        self.num_classes = num_classes
        
        self.file_list = sorted(
            [os.path.join(precomputed_dir, f) for f in os.listdir(precomputed_dir) if f.endswith('.npy')],
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]) if os.path.basename(x).split('_')[1].split('.')[0].isdigit() else 0
        )
        
        # Load seluruh label dan subject_id ke RAM untuk filtering dan splitting
        self.labels = []
        self.subject_ids = []
        self.valid_indices = []
        
        print(f"Memuat {len(self.file_list)} sampel graf dari {precomputed_dir}...")
        for idx, filepath in enumerate(self.file_list):
            data = np.load(filepath, allow_pickle=True).item()
            lbl = int(data["label"])
            sub = data.get("subject_id", "unknown")
            
            if lbl < self.num_classes:
                self.valid_indices.append(idx)
                self.labels.append(lbl)
                self.subject_ids.append(sub)
                
        print(f"✅ {len(self.valid_indices)} sampel valid berhasil dimuat ({self.num_classes} kelas)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        filepath = self.file_list[actual_idx]
        data = np.load(filepath, allow_pickle=True).item()
        
        graph_arr = data["graph"]  # Format: [Frames=16, Nodes=468, Channels=7] atau [Channels=7, Frames=16, Nodes=468]
        if graph_arr.shape[0] == 16 and graph_arr.shape[2] == 7:
            # Transpose ke [Channels=7, Frames=16, Nodes=468]
            graph_arr = np.transpose(graph_arr, (2, 0, 1))
            
        tensor_graph = torch.tensor(graph_arr, dtype=torch.float32)
        tensor_label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_graph, tensor_label


if os.path.exists(PRECOMPUTED_DIR) and len(os.listdir(PRECOMPUTED_DIR)) > 0:
    fast_full_dataset = PrecomputedGraphDataset(PRECOMPUTED_DIR, num_classes=NUM_CLASSES)
else:
    print(f"⚠️ Folder {PRECOMPUTED_DIR} belum tersedia. Silakan jalankan pra-komputasi offline terlebih dahulu.")