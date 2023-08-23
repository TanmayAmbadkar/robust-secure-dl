import torch
import numpy as np
from utils.attack import zoo_attack

def test_defense(model, defense, data_loader, num_images):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    true_attacked = []
    predicted_attacked = []
    predicted_proba = []
    total = 0
    for i, (images, labels) in enumerate(data_loader):
        
        images, labels = images.to(device), labels.to(device)
        activations, pred_label = model(images)
        if pred_label.item() != labels.item():
            continue

        total += 1
        
        #Checking defense on unattacked images
        
        true_attacked.append(0)
        if defense.predict(activations.cpu().numpy().reshape(1,-1)) == pred_label.item():
            predicted_attacked.append(0)
        else:
            predicted_attacked.append(1)
        
        predicted_proba.append(1 - defense.predict_proba(activations.cpu().numpy().reshape(1,-1))[0,pred_label.item()])
        
        
        #checking defense on attacked images
        
        adv_image = zoo_attack(network=model.resnet, image=images, label=labels)
        adv_image = adv_image.to(device)
        activation, label = model(adv_image)
        
        
        
        if defense.predict(activation.cpu().numpy().reshape(1,-1)) == label.cpu().numpy():
            predicted_attacked.append(0)
        else:
            predicted_attacked.append(1)
            
        predicted_proba.append(1 - defense.predict_proba(activation.cpu().numpy().reshape(1,-1))[0,label.item()])
        true_attacked.append(1)

        if total >= num_images:
            break
    
    return true_attacked, predicted_attacked