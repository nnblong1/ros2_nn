import csv
import glob
import os
file = "/home/wicom/uam_results/case_b_combined_ff_narrow_sweep_20260607_150103"
out = os.environ["file"]

print("config,verdict_counts,rate,angle,xy_mean,xy_max,n_hat,ff_max,residual")
for path in sorted(glob.glob(out + "/runs/*/combined_amp080/*_median.csv")):
    name = path.split("/runs/")[1].split("/")[0]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        print(
            name,
            r["verdict_counts"],
            r["median_rate_err_rms_radps"],
            r["median_angle_rms_deg"],
            r["median_xy_mean_m"],
            r["median_xy_max_m"],
            r["median_n_hat_norm_rms"],
            r["median_arm_ff_max_nm"],
            r["median_tau_residual_rms_nm"],
            sep=","
        )
