if __name__ == "__main__":

    from src.pipelines.train_pipeline import train_pipeline
    from src.services.controllers.inference_controller import infere
    from src.services.controllers.evaluate_controller import evaluate

    print("Type 'train' to train the models, 'evaluate' to evaluate the models or 'infere' to infere the result")
    OPERATION = input()

    TRAINING_DATA_DIR = "resources/MRI/Training"
    TESTING_DATA_DIR = "resources/MRI/Testing"
    IMAGE_PATH = "resources/MRI/Testing/meningioma_tumor/image.jpg"

    EPOCHS = 50
    LEARNING_RATE = 0.001
    MIN_ACCURACY = 0.9
    NUM_WORKERS = 2
    VERBOSE = False


    if OPERATION == "train":
        train_pipeline(TRAINING_DATA_DIR, TESTING_DATA_DIR, EPOCHS, LEARNING_RATE, MIN_ACCURACY, NUM_WORKERS, VERBOSE)
    elif OPERATION == "evaluate":
        evaluate(TESTING_DATA_DIR, NUM_WORKERS)
    elif OPERATION == "infere":
        infere(IMAGE_PATH)
    else:
        print("Unknown command, try again")