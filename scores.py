import pandas as pd

file_path = "COSE_SupUnsup_Unknowns_V1.xlsx"
sheet_name = "IDS_Scores"
df = pd.read_excel(file_path, sheet_name=sheet_name)

filtered_df = df[(df['datasetName'] == 'CICIDS17') & (df['experimentTag'] != 'Full') & (df['meta'] == 'Regular')]

averages = filtered_df.groupby(['classifierName', 'meta'])['rec unk'].mean()

sorted_averages = averages.sort_values(ascending=False)

for (classifier, meta), avg_rec_unk in sorted_averages.items():
    individual_scores = filtered_df[(filtered_df['classifierName'] == classifier) & (filtered_df['meta'] == meta)][['rec unk', 'experimentTag']].values
    individual_scores_str = ", ".join([f"{tag}: {score:.4f}" for score, tag in individual_scores])
    print(f"Classifier: {classifier} (Meta: {meta}), Average rec unk: {avg_rec_unk:.4f}, Individual Scores: [{individual_scores_str}]")
