# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torchvision import datasets, models, transforms

def set_parameter_requires_grad(model, n_resblock_finetune):
    assert n_resblock_finetune in (0, 1, 2, 3, 4, 5)
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        condition = (n_resblock_finetune >= 1 and 'layer4' in name) or (n_resblock_finetune >= 2 and 'layer3' in name) or \
                    (n_resblock_finetune >= 3 and 'layer2' in name) or (n_resblock_finetune >= 4 and 'layer1' in name) or \
                    (n_resblock_finetune >= 5)

        if condition:
            param.requires_grad = True

    for name, param in model.named_parameters():
        if 'bn' in name:
            param.requires_grad = False

def initialize_model(model_name, n_resblock_finetune, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, n_resblock_finetune)
        feature_size = model_ft.fc.in_features
        input_size = 224
    elif model_name == 'resnet34':
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, n_resblock_finetune)
        feature_size = model_ft.fc.in_features
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size, feature_size





