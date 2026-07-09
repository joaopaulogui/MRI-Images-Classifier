if __name__ == "__main__":

    from src.utils.test_data_augmentations import test_data_augmentation
    from src.pipelines.train_pipeline import train_pipeline
    from src.pipelines.evaluate_pipeline import evaluate_pipeline

    print("Type 'train' to train the models, 'evaluate' to evaluate the models")
    OPERATION = input()

    TRAINING_DATA_DIR = "resources/Kaggle/Training"
    TESTING_DATA_DIR = "resources/Kaggle/Testing"

    EPOCHS = 60
    LEARNING_RATE = 0.00001
    MIN_ACCURACY = 0.9
    EARLY_STOPPING_PATIENCE = 20
    REDUCE_LR_PATIENCE = 7
    NUM_WORKERS = 2
    VERBOSE = False


    if OPERATION == "train":
        train_pipeline(TRAINING_DATA_DIR, TESTING_DATA_DIR, EPOCHS, LEARNING_RATE, MIN_ACCURACY, NUM_WORKERS, VERBOSE, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE)
    elif OPERATION == "evaluate":
        evaluate_pipeline(TESTING_DATA_DIR, NUM_WORKERS)
    elif OPERATION == "test_data_augmentation":
        test_data_augmentation()
    else:
        print("Unknown command, try again")