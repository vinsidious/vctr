import numpy as np
import torch
import torch.nn.functional as F


class InferenceHandler:
    def __init__(self, model, history_size=100):
        self.model = model
        self.history_size = history_size
        self.logits_history = []
        self.targets_history = []

    def update_history(self, logits, targets):
        self.logits_history.append(logits)
        self.targets_history.append(targets)

        if len(self.logits_history) > self.history_size:
            self.logits_history.pop(0)
            self.targets_history.pop(0)

    def run_inference(self, inputs, targets):
        logits = self.model(inputs)

        # Update history
        self.update_history(logits, targets)

        # Determine the number of classes and examples
        num_classes = logits.shape[-1]
        num_examples = logits.shape[0]

        # Initialize the new predictions tensor on the GPU
        labels = torch.zeros(num_examples, device=inputs.device).to('mps')

        # Calculate the decision thresholds for each class based on historical logits and targets
        thresholds = torch.zeros(num_classes).to('mps')
        for i in range(num_classes):
            logits_for_class = []
            for logits, target in zip(self.logits_history, self.targets_history):
                logits_flat = logits.view(-1, num_classes)
                target_flat = target.view(-1)
                logits_for_class.extend(logits_flat[target_flat == i].tolist())
            if logits_for_class:
                thresholds[i] = np.percentile(logits_for_class, 100 - (100 / num_classes))

        # Make predictions using the decision thresholds
        for i in range(num_examples):
            label = torch.argmax(logits[i] - thresholds)
            labels[i] = label

        # Return the labels and logits
        return labels.cpu().numpy()


def run_inference(model, inputs, targets):
    logits = model(inputs)
    labels = torch.argmax(logits, dim=1)
    return labels.cpu().numpy(), logits.cpu().numpy()
