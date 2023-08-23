def get_predictions(model, dataset, test_start_idx = 1000, test_start_idx, model_type = "dnn"):
    
    true_labels = []
    pred_labels = []
    for i, (image, label) in enumerate(test_set):
        if i < test_start_idx:
            continue
        true_labels.append(label)
        if model_type == "dnn":
            pred_labels.append(model(image.reshape(1,*data[0].shape).cuda())[1].item())
        if i == test_start_idx:
            break