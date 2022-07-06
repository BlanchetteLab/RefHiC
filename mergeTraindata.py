import sys
import pickle

with open(sys.argv[1], 'rb') as handle:
    dataParams, \
    X_train, Xs_train, y_label_train, \
    X_test, Xs_test, y_label_test, \
    X_val, Xs_val, y_label_val = pickle.load(handle)

with open(sys.argv[2], 'rb') as handle:
    dataParams2, \
    X_train2, Xs_train2, y_label_train2, \
    X_test2, Xs_test2, y_label_test2, \
    X_val2, Xs_val2, y_label_val2 = pickle.load(handle)

print(dataParams)

X_train = X_train+X_train2
Xs_train = Xs_train + Xs_train2
y_label_train = y_label_train + y_label_train2

X_test = X_test+X_test2
Xs_test = Xs_test + Xs_test2
y_label_test = y_label_test + y_label_test2

X_val = X_val + X_val2
Xs_val = Xs_val + Xs_val2
y_label_val = y_label_val + y_label_val2

with open(sys.argv[3], 'wb') as handle:
    pickle.dump((dataParams, X_train, Xs_train, y_label_train,
                 X_test, Xs_test, y_label_test,
                 X_val, Xs_val, y_label_val), handle, protocol=pickle.HIGHEST_PROTOCOL)
