from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

def score(y_true, y_pred, label=''):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    final_score = 2 * precision * recall / (precision + recall)
    print(label, 'recall:', round(recall, 4), ',precision:', round(precision, 4), ',F1 score:', round(final_score, 4))
    return recall, precision, final_score