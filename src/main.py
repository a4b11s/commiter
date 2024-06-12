from train import train_model
from test import test
import os

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # test()
    train_model()
