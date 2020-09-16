from preprocessing.utils import get_train_test_data
import sys

dataset = str(sys.argv[1])

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = get_train_test_data(dataset=dataset)

    print(X_train.shape)
    print(y_train.shape)

    print(X_test.shape)
    print(y_test.shape)
