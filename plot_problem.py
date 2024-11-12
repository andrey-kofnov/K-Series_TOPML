from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
import matplotlib.font_manager as font_manager
from matplotlib import cm
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

def plot_1d_problem(grid, K_ser_est, samples, my_path, problem_params):

    var_ = problem_params['variable']

    styles = ["-", "--", "--", "--", ":", ":"]

    font1 = font_manager.FontProperties(  # family='Comic Sans MS',
        weight='bold',
        style='normal', size=18)

    font2 = font_manager.FontProperties(  # family='Comic Sans MS',
        weight='bold',
        style='normal', size=24)

    n_bins = 100 if problem_params['discrete'] == False else len(set(samples))

    f = plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')
    from matplotlib.ticker import FormatStrFormatter

    if problem_params['name'] == 'PDP':
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    else:
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.plot(grid, K_ser_est, linestyle=styles[1], linewidth=5, color='red')
    plt.hist(samples, bins=n_bins, density=True, color='white', edgecolor='blue', linewidth=1.2)
    plt.legend(["K-series", var_ + " histogram"], prop=font1, frameon=False)
    plt.savefig(my_path + "/" + var_ + "_estimator_" + problem_params['name'] + ".jpeg")




def plot_2d_problem(grid, K_ser_est, my_path, problem_params):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    surf = ax.plot_surface(grid[0], grid[1], K_ser_est, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # ax.set_zlim(0.0, 6.5)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.ax.tick_params(labelsize=15)

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(fontsize=10, weight='bold')
    ax.set_xlabel('$X$', fontsize=20, rotation=160, weight='bold')
    ax.set_ylabel('$Y$', fontsize=20, weight='bold')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('$\hat{f}$', fontsize=20, rotation=0, weight='bold')
    try:
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
        for t in ax.zaxis.get_major_ticks(): t.label.set_weight('bold')
    except:
        pass
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.savefig(my_path + "/estimator_2D_" + problem_params['name'] + ".jpeg")




def plot_1d_dist(grid, K_ser_est, true_dist, my_path, problem_params):

    var_ = problem_params['variable']

    styles = ["-", "--", "--", "--", ":", ":"]

    font1 = font_manager.FontProperties(  # family='Comic Sans MS',
        weight='bold',
        style='normal', size=18)

    font2 = font_manager.FontProperties(  # family='Comic Sans MS',
        weight='bold',
        style='normal', size=24)

    f = plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')
    from matplotlib.ticker import FormatStrFormatter

    if problem_params['name'] == 'PDP':
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    else:
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.plot(grid, K_ser_est, linestyle=styles[1], linewidth=5, color='red', label = "K-series")
    plt.plot(grid, true_dist, linestyle=styles[3], linewidth=3, color='blue', label = problem_params['truth_name'])
    plt.legend(prop=font1, frameon=False)
    plt.savefig(my_path + "/" + var_ + "_estimator_" + problem_params['name'] + ".jpeg")
