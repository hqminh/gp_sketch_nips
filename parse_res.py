from utils.utility import *


def parse_res(res_file):
    with open(res_file) as f:
        res_dict = {}
        lines = f.readlines()
        while len(lines):
            seed = int(lines.pop(0))
            s = ''
            while ']' not in s:
                s += lines.pop(0).strip('\n')
            res_dict[seed] = eval(s)

        return res_dict


def plot_res(exp_set, lbl_set, prefix='', postfix='', fig_name='compare.png'):
    create_std_plot()
    for t, res_file in enumerate(exp_set):
        res_dict = parse_res(prefix + res_file + postfix)
        x = np.arange(11) * 10
        y = np.zeros(11)
        err = np.zeros(11)
        for seed in res_dict.keys():
            for i, item in enumerate(res_dict[seed]):
                y[i] += item['rmse'] / len(res_dict.keys())

        for seed in res_dict.keys():
            for i, item in enumerate(res_dict[seed]):
                err[i] += (item['rmse'] - y[i]) ** 2 / len(res_dict.keys())

        plt.errorbar(
            x, y, yerr=err ** 0.5,
            linestyle='--', marker='^',
            linewidth=2, markersize=12, label=lbl_set[t])
        plt.legend()
        plt.xticks(x)
        plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        plt.xlabel('No. Iterations')
        plt.ylabel('RMSE of CO concentration (ppm)')
        plt.savefig(
            prefix + fig_name, dpi=300
        )


def create_std_plot():
    plt.figure()
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def main():
    prefix = './results/gas_full/'
    postfix = '.res'
    exp_set_1 = ['vaegp_16', 'ssgp_16']
    lbl_set_1 = ['Revisited SSGP ($p=16$)', 'SSGP ($p=16$)']
    exp_set_2 = ['vaegp_32', 'ssgp_32']
    lbl_set_2 = ['Revisited SSGP ($p=32$)', 'SSGP ($p=32$)']
    exp_set_3 = ['vaegp_64', 'ssgp_64']
    lbl_set_3 = ['Revisited SSGP ($p=64$)', 'SSGP ($p=64$)']
    plot_res(exp_set_1, lbl_set_1, prefix, postfix, 'compare_16.png')
    plot_res(exp_set_2, lbl_set_2, prefix, postfix, 'compare_32.png')
    plot_res(exp_set_3, lbl_set_3, prefix, postfix, 'compare_64.png')


if __name__ == '__main__':
    main()
