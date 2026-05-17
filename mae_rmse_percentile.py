import json
import math
from pathlib import Path

# =====================================================================
# CONFIGURATION
# =====================================================================
TARGET_DIR = Path("mae_overall")

def process_52cm_runs():
    if not TARGET_DIR.exists():
        print(f"[!] Directory '{TARGET_DIR}' not found. Please ensure it is in the current path.")
        return

    runs_data = []
    
    # =================================================================
    # 1. COLLECT DATA
    # =================================================================
    for run_dir in TARGET_DIR.iterdir():
        if not run_dir.is_dir(): 
            continue
        
        json_path = run_dir / f"{run_dir.name}_evaluation_report.json"
        if not json_path.exists(): 
            continue
            
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        dist_acc = data.get("distance_accuracy", {})
        
        # Ensure the run didn't fail and actually has samples
        if "MAE_m" not in dist_acc or "sample_count" not in dist_acc:
            continue
            
        runs_data.append({
            "run_name": data.get("run_name", run_dir.name),
            "MAE_m": dist_acc["MAE_m"],
            "RMSE_m": dist_acc["RMSE_m"],
            "p95": dist_acc["p95_error_m"],
            "samples": dist_acc["sample_count"]
        })

    if not runs_data:
        print("[!] No valid evaluation reports found.")
        return

    # =================================================================
    # 2. CALCULATE OVERALL METRICS (Weighted)
    # =================================================================
    total_samples = sum(run["samples"] for run in runs_data)
    
    # Overall MAE: Weighted Average
    sum_weighted_mae = sum(run["MAE_m"] * run["samples"] for run in runs_data)
    overall_mae = sum_weighted_mae / total_samples
    
    # Overall RMSE: Weighted Mean of Squares, then Square Root
    sum_weighted_mse = sum((run["RMSE_m"] ** 2) * run["samples"] for run in runs_data)
    overall_rmse = math.sqrt(sum_weighted_mse / total_samples)
    
    # Overall p95: Safe Upper Bound vs Approximation
    approx_p95 = sum(run["p95"] * run["samples"] for run in runs_data) / total_samples
    max_p95 = max(run["p95"] for run in runs_data)

    # =================================================================
    # 3. PRINT REPORT
    # =================================================================
    print(f"\n{'='*70}")
    print(" OVERALL METRICS FOR 52cm STATIC OBSTACLE EXPERIMENT")
    print(f"{'='*70}")
    print(f"Total Runs Processed : {len(runs_data)}")
    print(f"Total Frames/Samples : {total_samples}")
    print(f"{'-'*70}")
    
    # Showing values in both meters (m) and centimeters (cm) for clarity
    print(f"Overall MAE          : {overall_mae:.5f} m   ({overall_mae * 100:.2f} cm)")
    print(f"Overall RMSE         : {overall_rmse:.5f} m   ({overall_rmse * 100:.2f} cm)")
    print(f"Overall p95 (Max)    : {max_p95:.5f} m   ({max_p95 * 100:.2f} cm)  <-- Safe Upper Bound")
    print(f"Overall p95 (Approx) : {approx_p95:.5f} m   ({approx_p95 * 100:.2f} cm)  <-- Weighted Avg")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    process_52cm_runs()