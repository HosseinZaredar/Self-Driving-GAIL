import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')


# exponential moving average
def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


df_1 = pd.read_csv('runs/run-train-bc-gail_1_1662968645-tag-charts_episodic_return_disc.csv')
df_2 = pd.read_csv('runs/run-train-bc-gail_1_1662978164-tag-charts_episodic_return_disc.csv')
df_2 = df_2.loc[74_000 <= df_2['Step']]
df = pd.concat([df_1, df_2])
df = df.loc[df['Step'] <= 99_900]

plt.figure()
plt.plot(df['Step'].tolist(), df['Value'].tolist(), alpha=0.25, color='tab:blue')
plt.plot(df['Step'].tolist(), smooth(df['Value'].tolist(), 0.9), color='tab:blue')
plt.axhline(y=0.0, color='black', linestyle='--')
plt.ylim((-20, 20))
plt.xlabel('Environment Interactions')
plt.ylabel('Episodic Return')

plt.show()
