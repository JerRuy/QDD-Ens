from torch.autograd import Variable
import torch
import math
import numpy as np

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = v.data.cpu().numpy()
        # print(v)
        #v0 = v[0]
    sf = math.ceil(math.log2(v+1e-12))
    return sf

def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / (delta + 1e-12) + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input0, bits)
    v = torch.exp(v) * s
    return v

def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = linear_quantize(input0, sf, bits)
    v = torch.exp(v) * s
    return v

def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()

    if isinstance(min_val, Variable):
        # max_val = float(max_val.data.cpu().numpy()[0])
        # min_val = float(min_val.data.cpu().numpy()[0])
        max_val = float(max_val.data.cpu().numpy())
        min_val = float(min_val.data.cpu().numpy())

    input_rescale = (input - min_val) / (0.000000001 + max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n

    v =  v * (max_val - min_val) + min_val
    return v

def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input) # [-1, 1]
    input_rescale = (input + 1.0) / 2 #[0, 1]
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1 # [-1, 1]

    v = 0.5 * torch.log((1 + v) / (1 - v)) # arctanh
    return v


###########################################
###########################################
###########################################

def loadweight(net, path):
    model_dict = net.state_dict()
    checkpoint = torch.load(path)

    for k, v in checkpoint.items():
        name = k
        model_dict[name] = v
    net.load_state_dict(model_dict)
    return net


def quant_weight_dict(backbone, net, old_parameter_mask, new_parameter_mask, Q1, Q2, Q3, Q0):
    new_weight_dict = {}
    print('quanting')
    for k, v in net.state_dict().items():

        if len(v.size()) > 0:

            dev = v.device
            old_weight = v.data.cpu().numpy()

            if k.startswith(backbone+'1'):
                quant_weight = log_minmax_quantize(v, Q1).data.cpu().numpy()
                new_weight = np.where(new_parameter_mask[k], old_weight, quant_weight)
                new_weight = np.where(old_parameter_mask[k], new_weight, old_weight)

            elif k.startswith(backbone+'2'):
                quant_weight = log_minmax_quantize(v, Q2).data.cpu().numpy()
                new_weight = np.where(new_parameter_mask[k], old_weight, quant_weight)
                new_weight = np.where(old_parameter_mask[k], new_weight, old_weight)

            elif k.startswith(backbone+'3'):
                quant_weight = log_minmax_quantize(v, Q3).data.cpu().numpy()
                new_weight = np.where(new_parameter_mask[k], old_weight, quant_weight)
                new_weight = np.where(old_parameter_mask[k], new_weight, old_weight)
            else:
                quant_weight = log_minmax_quantize(v, Q0).data.cpu().numpy()
                new_weight = np.where(new_parameter_mask[k], old_weight, quant_weight)
                new_weight = np.where(old_parameter_mask[k], new_weight, old_weight)

            new_weight = torch.from_numpy(new_weight).to(dev)
            new_weight_dict[k] = new_weight

        else:
            new_weight_dict[k] = v

    return new_weight_dict


def update_weight(net, old_weight_dic, parameter_mask):  # for train every epoch
    new_weight = {}
    # print('quanting')
    for k, v in net.state_dict().items():
        if len(v.size()) > 0:
            dev = v.device
            old_weight = old_weight_dic[k].data.cpu().numpy()
            weight = v.data.cpu().numpy()
            weight = np.where(parameter_mask[k], weight, old_weight)
            weight = torch.from_numpy(weight).to(dev)
            new_weight[k] = weight
        else:
            new_weight[k] = v
    return new_weight


def two_sage_partition(net, old_parameter_mask, iterr, quant):
    new_parameter_mask = {}
    cnt = 0
    # print(quant[iterr-1])
    # print(np.sum(quant[:iterr-1]))
    for k, v in net.state_dict().items():
        cnt += 1
        mask = old_parameter_mask[k]    #quant map, ture: no quant , false: already quant
        weight = v.cpu().numpy()
        weight_no_inq = weight[mask]
        weight_2 = weight * mask        #same spatial dimension

        # print('1',np.shape(weight_no_inq))
        # print('2',np.shape(weight_2))
        quantratio = quant[iterr-1]*2 + np.sum(quant[:iterr-1])
        if quantratio < 1:
            median = np.percentile(abs(weight_no_inq), 100*(1-quantratio) )  # find the top k*Q% pivot
        else:
            median = -1

        new_mask = np.where(abs(weight_2) > median, False,mask)  # pruning-inspired matrix, ture: top k*Q% weights
        half = np.random.randint(0, 2, np.shape(new_mask))       # construct random matrix
        new_mask0 = np.where(half == 1, new_mask, True)          # merge pruning-inspired matrix and random matrix
        new_mask1 = np.where(new_mask0 == 0, new_mask0, mask)    # merge new quant matrix and old quant matrix
        new_parameter_mask[k] = new_mask1

        if cnt == 1:
            print('rate:', np.sum(new_mask1 == False) / mask.size)
    return old_parameter_mask, new_parameter_mask


