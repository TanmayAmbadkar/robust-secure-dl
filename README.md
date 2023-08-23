# Robust and Secure Deep Learning

1. Zoo Attack - Zeroth Order Optimization Attack involves adding controlled noise to an input image to induce a mis-classification during test time. Access to a trained model is necessary because the ZOO attack uses gradient descent using the final logits of the neural network. 

2. Test-time evasion detection - We need to create a detector that can identify if an image has been attacked during test time. Using a KNN classifier, we set a threshold on the number of votes an image receives, if the highest voted class is not equal to the predicted class, we can say that time image is attacked. The AUROC achieved with this method is 0.99

3. Backdoor poisoning - Backdoor poisoning involves a single source-target class pair, where a small fraction of samples (less tha 10%) in the training set are poisoned with a local Imperceptible pattern that induces a mis-classification. This reveals the fragility of deep learning based classifiers.

4. IPTRED - Imperceptible Post Training Reverse Engineering Defense is a method to reverse engineer the backdoor pattern used to poison the dataset. In this method, all possible source-target pairs are considered, where a source image is poisoned using a gradient descent optimised technique to induce a misclassification. Once it is run on all pairs, the real source-target pair can be detected when the estimated pattern is seen. 


The reports and the code are part of the repository