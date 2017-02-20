import numpy as np

def record_label(label_pred,label_true):
    # Initialize output
    output = np.zeros((label_true.shape[1],label_pred.shape[1]))

    # Turn prob-label into num-label
    num_pred = np.argmax(label_pred, 1)
    num_true = np.argmax(label_true, 1)

    # Change num of the table
    for i in range(label_true.shape[0]):
        output[num_pred[i]][num_true[i]] += 1

    return output


def record_error(label_pred, label_true, label_num):
    # Turn prob-label into num-label
    num_pred = np.argmax(label_pred, 1)
    num_true = np.argmax(label_true, 1)

    # Initialize output
    output = []
    tem = True

    # Change num of the table
    for i in range(label_true.shape[0]):
        if not num_pred[i] == num_true[i]:
            # Find if the list is empty
            if tem:
                output = [[label_num + i + 1, num_pred[i] + 1, num_true[i] + 1]]
                tem = False
            else:
                output = np.append(output, [[label_num + i + 1, num_pred[i] + 1, num_true[i] + 1]], axis=0)

    return output