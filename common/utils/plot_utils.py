import argparse
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# styling
# plt.style.use('seaborn-dark')
# matplotlib.rcParams.update({'font.size': 22})

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
COLORS = ["#95d0ff", "#966bff", "#ff6ad5", "#ffa58b", "#ff6a8b"]


# sns.color_palette('bright', 6)


def get_args():
    parser = argparse.ArgumentParser(description='Domain Randomization Driver')
    parser.add_argument('--environment', type=str,
                        choices=['lunar', 'pusher', 'ergo', 'ergosix', 'lunar2', 'lunarbootstrap'])
    parser.add_argument('--filter', type=float)
    return parser.parse_args()


def get_config(environment):
    if environment == 'lunar':
        return {
            'metrics': ['ref_learning_curve_{}', 'hard_learning_curve_{}', 'rand_learning_curve_{}'],
            'solved': 200,
            'xlim': (7.5, 20.0),
            'ylim': (0, 330),
            'start_index': 0,
            'environment': environment,
            # 'labels': ['baseline', 'UDR', 'oracle', 'ADR (ours)'],
            'labels': ['Oracle', 'Baseline', 'UDR', 'ADR (ours)'],
            'title': 'Generalization Results (LunarLander)',
            # 'title': 'Oracle vs. UDR (LunarLander)',
            'dimensions': 1,
            'colors': COLORS,
            'legend_loc': 'lower right',
            'x_label': 'Main Engine Strength (MES)',
            'y_label': 'Average Reward'
        }
    elif environment == 'lunar2':
        return {
            'metrics': ['ref_learning_curve_{}', 'hard_learning_curve_{}'],
            'solved': 200,
            'xlim': (7.5, 20.0),
            'ylim': (-100, 330),
            'start_index': 0,
            'environment': environment,
            'labels': ['$Baseline$', '$UDR$', '$ADR (ours)$'],
            'title': ['Learning Curve (LL), Reference Env.', 'Learning Curve (LL), Hard Env.'],
            'dimensions': 1,
            'colors': [COLORS[1], COLORS[2], COLORS[0]],
            'legend_loc': 'best',
            'x_label': 'Main Engine Strength (MES)',
            'y_label': 'Average Reward'
        }
    elif environment == 'lunarbootstrap':
        return {
            'metrics': ['ref_learning_curve_{}'],
            'solved': 200,
            'xlim': (7.5, 11),
            'ylim': (-150, 330),
            'start_index': 0,
            'environment': environment,
            'labels': ['$ADR(boostrapped)$', '$ADR(original)$'],
            'title': ['Bootstrapped ADR (LL)'],
            'dimensions': 1,
            'colors': [COLORS[1], COLORS[0]],
            'legend_loc': 'lower right',
            'x_label': 'Main Engine Strength (MES)',
            'y_label': 'Average Reward'
        }
    elif environment == 'pusher':
        return {
            'metrics': ['ref_final_dists_{}', 'hard_final_dists_{}'],
            'solved': 0.35,
            'xlim': (0, 1.0),
            'ylim': (0.1, 0.7),
            'start_index': 0,
            'environment': environment,
            'labels': ['$UDR$', '$ADR (ours)$'],
            'title': ['Learning Curve (Pusher), Reference Env.', 'Learning Curve (Pusher), Hard Env.'],
            'dimensions': 2,
            'colors': [COLORS[2], COLORS[0]],
            'legend_loc': 'upper right',
            'x_label': 'Agent Timesteps',
            'y_label': 'Average Final Distance to Goal'
        }

    elif environment == 'ergo':
        return {
            'metrics': ['ref_final_dists_{}', 'hard_final_dists_{}'],
            'solved': None,
            'xlim': (0, 1.0),
            'ylim': (0, 0.2),
            'start_index': 0,
            'environment': environment,
            'labels': ['$UDR$', '$ADR (ours)$'],
            'title': ['Learning Curve (Ergo), Reference Env.', 'Learning Curve (Ergo), Hard Env.'],
            'dimensions': 8,
            'colors': [COLORS[2], COLORS[0]],
            'legend_loc': 'upper right',
            'x_label': 'Agent Timesteps',
            'y_label': 'Average Final Distance to Goal'
        }


def gen_plot(config, file_path, data, title=None, learning_curve=False):
    plt.figure(figsize=(6, 5))

    plt.title(config['title'] if not title else title)
    plt.xlabel(config['x_label'])
    plt.ylabel(config['y_label'])

    plt.ylim(*config['ylim'])
    if config['solved']:
        # plt.axhline(config['solved'], color=COLORS[4], linestyle='--', label='$[Solved]$') # only for figure 1
        plt.axhline(config['solved'], color=COLORS[3], linestyle='--', label='$[Solved]$')

    # colors = config['colors'][::-1][1:] # only for figure 1
    colors = config['colors']
    for i, entry in enumerate(data):
        timesteps, averaged_curve, sigma, convergence = entry
        sns.lineplot(timesteps,
                     averaged_curve,
                     c=colors[i],
                     label=config['labels'][i])
        if convergence is not None:
            plt.plot([timesteps[-1], timesteps[-1] + 0.5],
                     [averaged_curve.values[-1], averaged_curve.values[-1]],
                     color=colors[i],
                     linestyle='--')

        plt.fill_between(x=timesteps,
                         y1=averaged_curve + sigma,
                         y2=averaged_curve - sigma,
                         facecolor=colors[i],
                         alpha=0.1)
    if learning_curve:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.legend(loc=config['legend_loc'], frameon=True, framealpha=0.5)
    plt.grid(b=False)

    # plt.show()

    plt.savefig(fname=file_path,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()
