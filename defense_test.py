import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from CNN.resnet import ResNet18
from load_data import load_data
import pdb
import torchvision.utils as vutils


training_set, test_set = load_data(data='mnist')

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