import pandas as pd

df = pd.read_csv("output/episodes_post_ib.csv")
count = (df["failure_count_at_start"] == 1).sum()
print(count)