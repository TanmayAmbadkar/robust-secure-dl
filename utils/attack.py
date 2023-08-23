import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def loss_fn(logits, label):
    
    log_probs = F.log_softmax(logits, dim=1).reshape(-1,)
    label_log_prob = log_probs[label].item()
    log_probs[label] = -1e8
    max_log_prob = log_probs.max()
    return max(label_log_prob - max_log_prob, 0)


def calc_gradient(network, image, label, constant, h):
    
    gradient = loss_fn(network(image+constant), label) - loss_fn(network(image-constant), label)
    gradient = gradient/h
    return gradient


def zoo_attack(network, image, label):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param label: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    eta = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10**-8
    alpha = 0.01
    M = torch.zeros(*image.shape).to(device)
    T = torch.zeros(*image.shape).to(device)
    v = torch.zeros(*image.shape).to(device)
    h = 0.001
    f = loss_fn(network(image), label)
    iter_= 0
    while f>0:
    # for _ in range(1000):
        iter_+=1
        # print("------------------------")
        # print("iter : ", iter_)
        # print(f"\rloss: {f:0.2f}", end = "")
        # randomly pick a coordinate
        e = torch.zeros((1, 1, 32, 32))
        r = np.random.randint(0, 31-3)
        c = np.random.randint(0, 31-3)
        e[0, 0, r:r+3, c:c+3] = h
        e = e.to(device)

        # # get the gradient of the loss function
        # f1 = calc_f1(network, image+e,label,)
        # f2 = calc_f1(network, image-e,label,)
        # gradient = (f1-f2)/(2*h)
        
        gradient = calc_gradient(network, image, label, e, h)
        
        # pdb.set_trace()
        # gradient = 0
        T[0, 0, r:r+3, c:c+3] = T[0, 0, r:r+3, c:c+3]+1
        M[0, 0, r:r+3, c:c+3] = beta1*M[0, 0, r:r+3, c:c+3] + (1-beta1)*gradient
        v[0, 0, r:r+3, c:c+3] = beta2*v[0, 0, r:r+3, c:c+3] + (1-beta2)*gradient**2
        M_hat = M[0, 0, r:r+3, c:c+3]/(1-beta1**T[0, 0, r:r+3, c:c+3])
        v_hat = v[0, 0, r:r+3, c:c+3]/(1-beta2**T[0, 0, r:r+3, c:c+3])
        
        image[0, 0, r:r+3, c:c+3] = image[0, 0, r:r+3, c:c+3] - eta*M_hat/(v_hat**0.5 + epsilon)
        f = loss_fn(network(image), label)
        # convergence = 

    # pdb.set_trace()
    return image