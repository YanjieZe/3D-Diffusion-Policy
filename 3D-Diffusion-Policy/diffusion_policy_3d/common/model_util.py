from termcolor import cprint

def print_params(model):
    """
    Print the number of parameters in each part of the model.
    """    
    # 定义一个字典来存储每个部分的参数数量
    params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())

    # 遍历所有参数
    for name, param in model.named_parameters():
        # 获取当前参数的部分名称，即第一个'.'之前的字符串
        part_name = name.split('.')[0]
        # 如果该部分名称不在字典中，则将其添加到字典中
        if part_name not in params_dict:
            params_dict[part_name] = 0
        # 将当前参数的数量加入到该部分的参数数量中
        params_dict[part_name] += param.numel()

    # 遍历所有部分，计算并打印总参数数量
    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Number of parameters: {all_num_param / 1e6:.4f}M', 'cyan')
    for part_name, num_params in params_dict.items():
        # print num (in M) and percentage
        cprint(f'   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})', 'cyan')
    cprint(f'----------------------------------', 'cyan')