class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_accuracy = None
        self.early_stop = False

    def __call__(self, metrics_dict):
        current_accuracy = metrics_dict["accuracy"]

        if self.best_accuracy is None:
            self.best_accuracy = current_accuracy
        elif current_accuracy < self.best_accuracy + self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_accuracy = current_accuracy
            self.counter = 0