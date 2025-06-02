import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from helper_code import load_text

# === Configuration ===
dataset_dirs = {
    "samitrop": "samitrop_output",
    "ptbxl": "ptbxl_output",
}

output_base = Path("dataset")  # Preserving original dataset/train and dataset/val structure
train_dir = output_base / "train"
val_dir = output_base / "val"
holdout_dir = Path("holdout_data")

# Create output directories
for d in [train_dir, val_dir, holdout_dir]:
    os.makedirs(d, exist_ok=True)

# === Symlink one record's files ===
def symlink_single_record(record, dest_dir, remove_label=False):
    for ext in ['.hea', '.dat', '.txt']:
        src_file = record.with_suffix(ext)
        if src_file.exists():
            dst_file = dest_dir / src_file.name  # Keep original filename (with _hr if present)
            try:
                if not dst_file.exists():
                    os.symlink(src_file.resolve(), dst_file)
            except FileExistsError:
                pass

            # Optional: remove label from .txt file (if your pipeline generates those)
            if remove_label and ext == '.txt':
                text = load_text(src_file)
                text['label'] = None
                with open(dst_file, 'w') as f:
                    f.write(str(text))

# === Parallel symlinking with progress ===
def symlink_records_parallel(records, dest_dir, remove_label=False, label=""):
    os.makedirs(dest_dir, exist_ok=True)
    total = len(records)
    print(f"[{label}] Starting symlink of {total} records to {dest_dir}...")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(symlink_single_record, record, dest_dir, remove_label): i
            for i, record in enumerate(records)
        }

        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"[{label}] Processed {i + 1}/{total} records")

    print(f"[{label}] Done.\n")

# === Main dataset split and processing ===
for dataset_name, dataset_path in dataset_dirs.items():
    dataset_path = Path(dataset_path)
    
    # Get all .hea files, strip extension to keep base name with _hr if present
    record_list = sorted([f.with_suffix('') for f in dataset_path.glob("*.hea")])
    random.shuffle(record_list)

    n = len(record_list)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_records = record_list[:n_train]
    val_records = record_list[n_train:n_train + n_val]
    test_records = record_list[n_train + n_val:]

    print(f"\nProcessing dataset: {dataset_name}")
    symlink_records_parallel(train_records, train_dir, label=f"{dataset_name} - train")
    symlink_records_parallel(val_records, val_dir, label=f"{dataset_name} - val")

    if dataset_name in ["samitrop", "ptbxl"]:
        symlink_records_parallel(test_records, holdout_dir, remove_label=True, label=f"{dataset_name} - holdout")
