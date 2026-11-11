import json, glob, sys
import pandas as pd
rows = []
for f in glob.glob('results/e4b_part*.json'):
    rows += json.load(open(f))
if not rows:
    sys.exit('no e4b results')
A = pd.DataFrame(rows)
pd.set_option('display.width', 220)
T = A.groupby(['prior', 'method'])[['log_sd_ratio', 'rms_log_sd_ratio', 'mean_dev', 'cov95_truth', 'rhat', 'rhat90']].median().round(3)
T.to_csv('results/e4b_table.csv')
print(T.to_string())
print(A.groupby(['problem', 'method']).rms_log_sd_ratio.median().unstack().round(3).to_string())
