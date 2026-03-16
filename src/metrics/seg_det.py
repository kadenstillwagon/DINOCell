import os
import numpy as np
import subprocess
import re
import cv2
import shutil
import random



def add_to_cell_tracking_challenge_format(gt_mask, pred_mask, output_path, seq_str, frame_idx):
    """Exports a single slice of GT and predicted mask in CTC style."""
    filename_gt_seg = f"{output_path}/{seq_str}_GT/SEG/man_seg{frame_idx:04d}.tif"
    filename_gt_tra = f"{output_path}/{seq_str}_GT/TRA/man_track{frame_idx:04d}.tif"
    filename_pred   = f"{output_path}/{seq_str}_RES/mask{frame_idx:04d}.tif"

    cv2.imwrite(filename_gt_seg, gt_mask.astype(np.uint16))
    cv2.imwrite(filename_gt_tra, gt_mask.astype(np.uint16))
    cv2.imwrite(filename_pred, pred_mask.astype(np.uint16))


def get_seg_det_metrics(gt_anns, model_pred_anns):
    output_path = "../cell-det-seg" # dataset formatted for the cell tracking challenge
    output_path = output_path + f'_{random.randint(1, 1000000000)}'
    os.makedirs(output_path, exist_ok=True)
    sequence_number = 1
    seq_str = f"{sequence_number:02d}"

    # Create subfolders
    for subdir in ["SEG", "TRA"]:
        os.makedirs(f"{output_path}/{seq_str}_GT/{subdir}", exist_ok=True)
    os.makedirs(f"{output_path}/{seq_str}_RES", exist_ok=True)

    # Write frames
    for i in range(len(gt_anns)):
        add_to_cell_tracking_challenge_format(gt_anns[i], model_pred_anns[i], output_path, seq_str, i)

    binary_path = "../cell-tracking-binaries"
    os.system(f'chmod +x {binary_path}/SEGMeasure')
    os.system(f'chmod +x {binary_path}/DETMeasure')

    def run_binary(binary_name):
        cmd = [f"{binary_path}/{binary_name}", output_path, f"{sequence_number:02d}", "04"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout.strip()
        # Look for pattern like "SEG measure: 0.761773" or "DET measure: 0.951793"
        match = re.search(r"([A-Z]+)\s+measure:\s*([0-9.]+)", stdout)
        if match:
            return float(match.group(2))
        else:
            print(f"Could not parse {binary_name} output:\n{stdout}")
            return None

    seg_score = run_binary("SEGMeasure")
    det_score = run_binary("DETMeasure")

    shutil.rmtree(output_path)

    return seg_score, det_score