def true_positive_rate(true, pred):
    return sum(pred[true == 1])/sum(true[true == 1])
    
def false_positive_rate(true, pred):
    return sum(pred[true == 0])/len(true[true == 0])