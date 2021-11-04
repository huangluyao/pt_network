import torch
import numpy as np
import torch.nn as nn



def get_ignore_bn_list(model):
    ignore_bn_list = []
    for k, m in model.named_modules():
        if m._get_name() == "Bottleneck":
            if m.add:
                ignore_bn_list.append(k.rsplit(".", 2)[0] + ".cv1.bn")
                ignore_bn_list.append(k + '.cv1.bn')
                ignore_bn_list.append(k + '.cv2.bn')

    return ignore_bn_list

def get_bn_weights(model, ignore_bn_list):
    # 显示bn的权重
    module_list = []
    module_bias_list = []
    for i, layer in model.named_modules():
        if "Head" in layer._get_name():
            break
        if isinstance(layer, nn.BatchNorm2d) and i not in ignore_bn_list:
            bnw = layer.state_dict()['weight']
            bnb = layer.state_dict()['bias']
            module_list.append(bnw)
            module_bias_list.append(bnb)

    size_list = [idx.data.shape[0] for idx in module_list]

    bn_weights = torch.zeros(sum(size_list))
    bnb_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in enumerate(size_list):
        bn_weights[index:(index + size)] = module_list[idx].data.abs().clone()
        bnb_weights[index:(index + size)] = module_bias_list[idx].data.abs().clone()
        index += size

    bn_weights = torch.zeros(sum(size_list))
    bnb_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in enumerate(size_list):
        bn_weights[index:(index + size)] = module_list[idx].data.abs().clone()
        bnb_weights[index:(index + size)] = module_bias_list[idx].data.abs().clone()
        index += size

    return bn_weights, bnb_weights


def get_model_list(model):
    model_list = {}
    ignore_bn_list = []

    for i, layer in model.named_modules():
        if layer._get_name() == "Bottleneck":
            if layer.add:
                ignore_bn_list.append(i.rsplit(".",2)[0]+".cv1.bn")
                ignore_bn_list.append(i + '.cv1.bn')
                ignore_bn_list.append(i + '.cv2.bn')
        elif "Head" in layer._get_name():
            break
        if isinstance(layer, nn.BatchNorm2d):
            if i not in ignore_bn_list:
                model_list[i] = layer
    model_list = {k:v for k,v in model_list.items() if k not in ignore_bn_list}
    return model_list, ignore_bn_list

def pruned_model_load_state(mask_bn_dict, pruned_model, model):

    changed_state = []
    old_model_dict = dict(model.named_modules())
    new_model_dict = dict(pruned_model.named_modules())
    former_to_map = pruned_model.former_to_map
    for pruned_layername, pruned_layer in new_model_dict.items():
        if isinstance(pruned_layer, nn.Conv2d):
            mask_name = pruned_layername[:-4] + "bn"
            if mask_name in former_to_map.keys():
                former = former_to_map[mask_name]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(mask_bn_dict[mask_name].detach().cpu().numpy()))
                    in_idx =  np.squeeze(np.argwhere(mask_bn_dict[former].detach().cpu().numpy()))
                    w = old_model_dict[pruned_layername].weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape)==3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w.clone()
                    changed_state.append(pruned_layername + ".weight")
                elif isinstance(former, list):
                    in_idx = []
                    orignin = [old_model_dict[i].weight.shape[0] for i in former]
                    for i in range(len(former)):
                        name = former[i]
                        tmp = [index for index in range(mask_bn_dict[name].shape[0]) if mask_bn_dict[name][index] ==1]
                        if i > 0 :
                            tmp = [k + sum(orignin[:i]) for k in tmp]
                        in_idx.extend(tmp)

                    out_idx = np.squeeze(np.argwhere(mask_bn_dict[mask_name].detach().cpu().numpy()))

                    w = old_model_dict[pruned_layername].weight.data[out_idx, :, :, :].clone()
                    pruned_layer.weight.data = w[:, in_idx, :, :].clone()
                    changed_state.append(pruned_layername + ".weight")
            elif "bbox_head" in mask_name:
                # 为了适用大多数算法，头部分，没有剪枝
                conv_names = pruned_layername.split(".")
                if conv_names[1] == "cls_convs" or conv_names[1] == "reg_convs":
                    if conv_names[-2] == "0":
                        former_name = "neck.out_convs.%s.bn" %( conv_names[-3])
                        in_idx = np.squeeze(np.argwhere(np.asarray(mask_bn_dict[former_name].cpu().numpy())))
                        pruned_layer.weight.data = old_model_dict[pruned_layername].weight.data[:, in_idx, :, :].clone()
                        if pruned_layer.bias is not None:
                            pruned_layer.bias.data = old_model_dict[pruned_layername].bias.data
                            changed_state.append(pruned_layername + ".bias")

                        changed_state.append(pruned_layername + ".weight")
                    else:
                        pruned_layer.weight.data = old_model_dict[pruned_layername].weight.data.clone()
                        if pruned_layer.bias is not None:
                            pruned_layer.bias.data = old_model_dict[pruned_layername].bias.data
                            changed_state.append(pruned_layername + ".bias")
                        changed_state.append(pruned_layername + ".weight")
                else:
                    pruned_layer.weight.data = old_model_dict[pruned_layername].weight.data.clone()
                    if pruned_layer.bias is not None:
                        pruned_layer.bias.data = old_model_dict[pruned_layername].bias.data
                        changed_state.append(pruned_layername + ".bias")
                    changed_state.append(pruned_layername + ".weight")
            else:
                out_idx = np.squeeze(np.argwhere(mask_bn_dict[mask_name].detach().cpu().numpy()))
                w = old_model_dict[pruned_layername].weight.data[out_idx, ...].clone()
                pruned_layer.weight.data = w.clone()
                changed_state.append(pruned_layername + ".weight")

        elif isinstance(pruned_layer, nn.BatchNorm2d):
            layer = old_model_dict[pruned_layername]
            if pruned_layername in mask_bn_dict.keys():
                out_idx = np.squeeze(np.argwhere(mask_bn_dict[pruned_layername].detach().cpu().numpy()))
                pruned_layer.weight.data = layer.weight.data[out_idx].clone()
                pruned_layer.bias.data = layer.bias.data[out_idx].clone()
                pruned_layer.running_mean = layer.running_mean[out_idx].clone()
                pruned_layer.running_var = layer.running_var[out_idx].clone()
            else:
                pruned_layer.weight.data = layer.weight.data.clone()
                pruned_layer.bias.data = layer.bias.data.clone()
                pruned_layer.running_mean = layer.running_mean.clone()
                pruned_layer.running_var = layer.running_var.clone()

            changed_state.append(pruned_layername + ".weight")
            changed_state.append(pruned_layername + ".bias")
            changed_state.append(pruned_layername + ".running_mean")
            changed_state.append(pruned_layername + ".running_var")
            changed_state.append(pruned_layername + ".num_batches_tracked")

        elif isinstance(pruned_layer, nn.GroupNorm):
            layer = old_model_dict[pruned_layername]
            pruned_layer.weight.data = layer.weight.data.clone()
            pruned_layer.bias.data = layer.bias.data.clone()
            changed_state.append(pruned_layername + ".weight")
            changed_state.append(pruned_layername + ".bias")
        elif pruned_layer._get_name() == "Scale":
            pruned_layer.scale.data = old_model_dict[pruned_layername].scale.data.clone()
            changed_state.append(pruned_layername+".scale")
        else:

            pass



    missing = [i for i in pruned_model.state_dict() if i not in changed_state]

    return missing
