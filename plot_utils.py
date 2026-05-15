def plot_by_name(plot_dict, name):
    plot_function, plot_args = plot_dict[name]
    plot_function(**plot_args)