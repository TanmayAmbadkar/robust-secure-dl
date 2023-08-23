import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from CNN.resnet import ResNet18
from load_data import load_data
import pdb
import torchvision.utils as vutils
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

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

#### KNN Defense creation

class IntermediateRep(nn.Module):
    
    def __init__(self, resnet, layer_to_extract):
        super(IntermediateRep, self).__init__()
        self.resnet = resnet
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.layer_to_extract = layer_to_extract
        
    def forward(self, x):
        
        resnet_out = torch.argmax(F.softmax(self.resnet(x), dim=1))
        for name, layer in self.resnet.named_children():
            x = layer(x)
            if name == self.layer_to_extract:
                break
        
        return x, resnet_out

intermediate = IntermediateRep(model, "layer1").cuda().eval()

intermediate_reps, true_labels, predicted_labels = [], [], []
for i, value in enumerate(test_set):
    if i == 1000:
        break
    data_point = value[0].cuda().reshape(1, *value[0].shape)
    with torch.no_grad():
        activations, label = intermediate(data_point)
    
    if label!=value[1]:
        continue
    intermediate_reps.append(activations.reshape(-1,).cpu().numpy())
    true_labels.append(value[1])
    predicted_labels.append(label.cpu().numpy())
    del data_point

intermediate_reps = np.array(intermediate_reps)
predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)

kmeans_models = []
for i in range(10):
    kmeans = KMeans(n_clusters = 10, random_state = 43)
    kmeans.fit(intermediate_reps[true_labels == i])
    kmeans_models.append(kmeans)
    

knn_X = []
knn_y = []

for i, kmeans in enumerate(kmeans_models):
    knn_X.append(kmeans_models[i].cluster_centers_)
    knn_y.append(np.array([i]*10))

knn_X = np.concatenate(knn_X, axis=0)
knn_y = np.concatenate(knn_y, axis=0)

knn_classifier = KNeighborsClassifier(n_neighbors = 3)
knn_classifier.fit(knn_X, knn_y)

true_attacked = []
predicted_attacked = []

for i in range(len(knn_X)):
    
    true_attacked.append(0)
    if knn_classifier.predict(intermediate_reps[i].reshape(1,-1)) == predicted_labels[i]:
        predicted_attacked.append(0)
        
    else:
        predicted_attacked.append(1)

print("finished KNN Defense")        
######



# todo note below is an example of getting the Z(X) vector in the ZOO paper

'''
z = model(image)

# if we consider just one image with size (1, 1, 32, 32)
# z.size() :   (1, 10)  10 elements are corresponding to classes

'''
# def max_index_excluding(inputs, exclude_index):
#     indices = torch.arange(inputs.size(0))
#     indices = indices[indices != exclude_index]
#     return indices[torch.argmax(inputs[indices])]
# def calc_f1(network, image,gt,):
#     #pdb.set_trace()
#     f1 = network(image)
#     f1 = F.log_softmax(f1, dim=1)
#     #f1 = torch.topk(f1, 2)
#     second_highest_prob_index = max_index_excluding(f1[0], gt.item())#fix this to a label? instead of changing each iteratio
#     #pdb.set_trace()
#     res = f1[0][gt.item()]-f1[0][second_highest_prob_index.item()]
#     #res = f1.values[0][gt.item()]-f1.values[0][second_highest_prob_index.item()]
#     return max(res.item(),0)

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


def zoo_attack(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    eta = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10**-8
    alpha = 0.01
    M = torch.zeros(*image.shape).to(device)
    T = torch.zeros(*image.shape).to(device)
    V = torch.zeros(*image.shape).to(device)
    h = 0.001
    f = loss_fn(network(image), t_0)
    iter_= 0
    while f>0:
    # for _ in range(1000):
        iter_+=1
        # print("------------------------")
        # print("iter : ", iter_)
        # print(f"\rloss: {f:0.2f}", end = "")
        # randomly pick a coordinate
        e = torch.zeros((1, 1, 32, 32))
        r = random.randint(0, 31-3)
        c = random.randint(0, 31-3)
        e[0, 0, r:r+3, c:c+3] = h
        e = e.to(device)

        # # get the gradient of the loss function
        # f1 = calc_f1(network, image+e,t_0,)
        # f2 = calc_f1(network, image-e,t_0,)
        # gradient = (f1-f2)/(2*h)
        
        gradient = calc_gradient(network, image, t_0, e, h)
        
        # pdb.set_trace()
        # gradient = 0
        T[0, 0, r:r+3, c:c+3] = T[0, 0, r:r+3, c:c+3]+1
        M[0, 0, r:r+3, c:c+3] = beta1*M[0, 0, r:r+3, c:c+3] + (1-beta1)*gradient
        V[0, 0, r:r+3, c:c+3] = beta2*v[0, 0, r:r+3, c:c+3] + (1-beta2)*gradient**2
        M_hat = M[0, 0, r:r+3, c:c+3]/(1-beta1**T[0, 0, r:r+3, c:c+3])
        v_hat = v[0, 0, r:r+3, c:c+3]/(1-beta2**T[0, 0, r:r+3, c:c+3])
        
        image[0, 0, r:r+3, c:c+3] = image[0, 0, r:r+3, c:c+3] - eta*M_hat/(v_hat**0.5 + epsilon)
        f = loss_fn(network(image), t_0)
        # convergence = 

    # pdb.set_trace()
    return image

def zoo_attack_newton(network, image, t_0):
    '''

    #todo you are required to complete this part
    :param network: the model
    :param image: one image with size: (1, 1, 32, 32) type: torch.Tensor()
    :param t_0: real label
    :return: return a torch tensor (attack image) with size (1, 1, 32, 32)
    '''
    eta = 0.01
    h = 0.0001
    f = calc_f1(network, image,t_0,)
    while f>0:
        print("Newton------------------------")
        # print("Ti: ", Ti)
        print("f: ", f)
        # randomly pick a coordinate
        e = torch.zeros((1, 1, 32, 32))
        r = random.randint(0, 31)
        c = random.randint(0, 31)
        e[0, 0, r:r+3, c:c+3] = h
        e = e.to(device)

        # get the gradient of the loss function
        f1 = calc_f1(network, image+e,t_0,)
        f2 = calc_f1(network, image-e,t_0,)
        gradient = (f1-f2)/(2*h)
        hessian = (f1-2*f+f2)/(h**2)
        if hessian<=0:
            delta = -eta*gradient
        else:
            delta = -eta*(gradient/hessian)
        image[0, 0, r:r+3, c:c+3] = image[0, 0, r:r+3, c:c+3] + delta
        f = calc_f1(network, image,t_0,)

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

    # adv_image = zoo_attack(network=model, image=images, target=target_label)
    # adv_image = zoo_attack_newton(network=model, image=images, t_0=labels)#zoo_attack(network=model, image=images, t_0=labels)
    adv_image = zoo_attack(network=model, image=images, t_0=labels)
    adv_image = adv_image.to(device)
    activation, label = intermediate(adv_image)
    if knn_classifier.predict(activation.cpu().numpy().reshape(1,-1)) == label.cpu().numpy():
        predicted_attacked.append(0)
    else:
        predicted_attacked.append(1)
    
    print(f"\r{total}", end="")
    true_attacked.append(1)
    
    
    
    #else:
    #    pdb.set_trace()
    # vutils.save_image(adv_image, './adv_images/'+str(total)+'_'+str(adv_pred.item())+'_'+str(labels.item())+'_image.png')


    if total >= num_image:
        break

print(classification_report(true_attacked, predicted_attacked))


print(confusion_matrix(true_attacked, predicted_attacked))