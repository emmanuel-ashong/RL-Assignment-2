import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob


def plot_results():
    config_map = {
        "full": "Full DQN",
        "naive": "Naive",
        "only_tn": "Only TN",
        "only_er": "Only ER"
    }

    all_data = []

    for keyword, label in config_map.items():
        pattern = f"**/*{keyword}*/**/*.csv"
        files = glob.glob(pattern, recursive=True)

        if not files:
            files = glob.glob(f"**/*{keyword}*.csv", recursive=True)

        for f in files:
            try:
                df = pd.read_csv(f)
                df.columns = [c.lower() for c in df.columns]

                if 'step' in df.columns and 'reward' in df.columns:
                    df['step_group'] = (df['step'] // 10000) * 10000
                    df = df.groupby('step_group')['reward'].mean().reset_index()
                    df.columns = ['step', 'reward']
                    df['reward'] = df['reward'].rolling(window=10, min_periods=1).mean()
                    df['configuration'] = label
                    all_data.append(df)
            except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue

    if not all_data:
        return

    df = pd.concat(all_data, ignore_index=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=df, x="step", y="reward", hue="configuration", linewidth=2.5)

    plt.title("DQN Performance on CartPole-v1: Ablation Study", fontsize=15, pad=15)
    plt.xlabel("Total Environment Steps", fontsize=12)
    plt.ylabel("Average Reward (Smoothed)", fontsize=12)
    plt.axhline(y=500, color='red', linestyle='--', alpha=0.3, label='Maximum Score')

    plt.xlim(0, 1000000)
    plt.ylim(0, 550)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    plt.savefig('dqn_ablation_results.png', dpi=300)


if __name__ == "__main__":
    plot_results()