import torch
import random
from CNN.resnet import ResNet18
from load_data import load_data
#import torch.backends.cudnn as cudnn
import pdb
import torchvision.utils as vutils
# load the mnist dataset (images are resized into 32 * 32)
training_set, test_set = load_data(data='mnist')

# define the model
model = ResNet18()#(dim=1)
# print("model: ", model)

# detect if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# load the learned model parameters
model.load_state_dict(torch.load('./model_weights/cpu_model.pth'))

model.to(device)
model.eval()

# todo note below is an example of getting the Z(X) vector in the ZOO paper

'''
z = model(image)

# if we consider just one image with size (1, 1, 32, 32)
# z.size() :   (1, 10)  10 elements are corresponding to classes

'''
def max_index_excluding(inputs, exclude_index):
    indices = torch.arange(inputs.size(0))
    indices = indices[indices != exclude_index]
    return indices[torch.argmax(inputs[indices])]
def calc_f1(network, image,gt,lg_sft_mx):
    #pdb.set_trace()
    f1 = network(image)
    f1 = lg_sft_mx(f1)
    #f1 = torch.topk(f1, 2)
    second_highest_prob_index = max_index_excluding(f1[0], gt.item())#fix this to a label? instead of changing each iteratio
    #pdb.set_trace()
    res = f1[0][gt.item()]-f1[0][second_highest_prob_index.item()]
    #res = f1.values[0][gt.item()]-f1.values[0][second_highest_prob_index.item()]
    return max(res.item(),0)

def zoo_attack(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    eta = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10**-8
    alpha = 0.01
    Mi = 0
    Ti = 0
    vi = 0
    h = 0.0001
    lg_sft_mx = torch.nn.LogSoftmax()
    lg_sft_mx.to(device)
    f = calc_f1(network, image,t_0,lg_sft_mx)
    while f>0.001:
        #print("------------------------")
        #print("Ti: ", Ti)
        #print("f: ", f)
        # randomly pick a coordinate
        e = torch.zeros((1, 1, 32, 32))
        r = random.randint(0, 31-3)
        c = random.randint(0, 31-3)
        e[0, 0, r:r+3, c:c+3] = h
        e = e.to(device)

        # get the gradient of the loss function
        f1 = calc_f1(network, image+e,t_0,lg_sft_mx)
        f2 = calc_f1(network, image-e,t_0,lg_sft_mx)
        gradient = (f1-f2)/(2*h)
        # pdb.set_trace()
        # gradient = 0
        Ti = Ti+1
        Mi = beta1*Mi + (1-beta1)*gradient
        vi = beta2*vi + (1-beta2)*gradient**2
        Mi_hat = Mi/(1-beta1**Ti)
        vi_hat = vi/(1-beta2**Ti)
        image = image - eta*Mi_hat/(vi_hat**0.5 + epsilon)
        f = calc_f1(network, image,t_0,lg_sft_mx)
        # convergence = 

    # pdb.set_trace()
    return image

# test the performance of attack
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])



total = 0
success = 0
num_image = 200 # number of images to be attacked

for i, (images, labels) in enumerate(testloader):
    target_label = get_target(labels)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = outputs.max(1)
    if predicted.item() != labels.item():
        continue

    total += 1

    #adv_image = zoo_attack(network=model, image=images, target=target_label)
    adv_image = zoo_attack(network=model, image=images, t_0=labels)
    adv_image = adv_image.to(device)
    adv_output = model(adv_image)
    _, adv_pred = adv_output.max(1)
    if adv_pred.item() != labels.item():
        success += 1
    #else:
    #    pdb.set_trace()
    print("success : ",success)
    vutils.save_image(adv_image, './adv_images/'+str(total)+'_'+str(adv_pred.item())+'_'+str(labels.item())+'_image.png')


    if total%10 == 0:
        print("success : ",success)
        print("total : ",total)
        print('success rate : %.4f'%(success/total))

    if total >= num_image:
        break

print('success rate : %.4f'%(success/total))