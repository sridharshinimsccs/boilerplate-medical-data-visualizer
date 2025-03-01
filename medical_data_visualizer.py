import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv("medical_examination.csv")

# 2. Add an 'overweight' column (BMI > 25 is overweight)
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (df['BMI'] > 25).astype(int)
df.drop(columns=['BMI'], inplace=True)

# 3. Normalize 'cholesterol' and 'gluc' (1 if normal, 0 otherwise)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Function to draw categorical plot
def draw_cat_plot():
    # 5. Create DataFrame for categorical plot
    df_cat = pd.melt(df, id_vars=["cardio"], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Draw the catplot
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat)

    # 8. Get the underlying matplotlib figure
    fig = g.fig  # Fix: Extracting figure from FacetGrid

    # 9. Save figure
    fig.savefig('catplot.png')
    return fig

# 10. Function to draw heat map
def draw_heat_map():
    # 11. Clean the data (filtering invalid measurements)
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15. Draw the heatmap
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, square=True, linewidths=0.5, cmap="coolwarm", ax=ax)

    # 16. Save figure
    fig.savefig('heatmap.png')
    return fig
