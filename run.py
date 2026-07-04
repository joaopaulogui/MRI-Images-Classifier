if __name__ == "__main__":

    from src.utils.test_data_augmentations import test_data_augmentation
    from src.pipelines.train_pipeline import train_pipeline
    from src.services.controllers.inference_controller import infere
    from src.pipelines.evaluate_pipeline import evaluate_pipeline

    print("Type 'train' to train the models, 'evaluate' to evaluate the models or 'infere' to infere the result")
    OPERATION = input()

    TRAINING_DATA_DIR = "resources/Kaggle/Training"
    TESTING_DATA_DIR = "resources/Kaggle/Training"
    IMAGE_PATH = "resources/Kaggle/Testing/meningioma_tumor/image.jpg"

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
    elif OPERATION == "infere":
        infere(IMAGE_PATH)
    elif OPERATION == "test_data_augmentation":
        test_data_augmentation()
    else:
        print("Unknown command, try again")