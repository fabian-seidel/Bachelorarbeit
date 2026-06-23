import inspect


def plot_by_name(plot_dict, name):
    plot_function, plot_args = plot_dict[name]
    plot_function(**plot_args)


def execute_for_param_cases(func, cases, arg_map=None, paths=None, **kwargs):
    paths = paths or {}
    arg_map = arg_map or {}
    func_params = inspect.signature(func).parameters
    case_keys = cases.keys()

    for vals in zip(*cases.values()):
        single_case = dict(zip(case_keys, vals))
        print(f'Executing case {vals}')
        run_paths = {k: v.format(**single_case) for k, v in paths.items()}
        matched_case_params = {}
        for param_name in func_params:
            dict_key = arg_map.get(param_name, param_name)
            if dict_key in single_case:
                matched_case_params[param_name] = single_case[dict_key]

        func(**kwargs, **matched_case_params, paths=run_paths)
