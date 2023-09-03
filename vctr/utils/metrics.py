import tensorflow as tf
from keras import backend as K
from keras.metrics import Metric


class PrecisionMacro(Metric):
    def __init__(self, name='prec', **kwargs):
        super(PrecisionMacro, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true, 'bool')
        y_pred = K.cast(K.round(y_pred), 'bool')

        true_positives = K.sum(K.cast(y_true & y_pred, 'float32'))
        false_positives = K.sum(K.cast(~y_true & y_pred, 'float32'))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        return precision


class RecallMacro(Metric):
    def __init__(self, name='rec', **kwargs):
        super(RecallMacro, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true, 'bool')
        y_pred = K.cast(K.round(y_pred), 'bool')

        true_positives = K.sum(K.cast(y_true & y_pred, 'float32'))
        false_negatives = K.sum(K.cast(y_true & ~y_pred, 'float32'))

        self.true_positives.assign_add(true_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return recall


class F1Macro(Metric):
    def __init__(self, name='f1', **kwargs):
        super(F1Macro, self).__init__(name=name, **kwargs)
        self.precision = PrecisionMacro()
        self.recall = RecallMacro()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        f1 = (
            2
            * (self.precision.result() * self.recall.result())
            / (self.precision.result() + self.recall.result() + K.epsilon())
        )
        return f1


class Accuracy(Metric):
    def __init__(self, name='acc', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.correct_predictions = self.add_weight(name='cp', initializer='zeros')
        self.total_predictions = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true, 'bool')
        y_pred = K.cast(K.round(y_pred), 'bool')

        correct_predictions = K.sum(K.cast(y_true == y_pred, 'float32'))

        self.correct_predictions.assign_add(correct_predictions)
        self.total_predictions.assign_add(K.cast(K.prod(K.shape(y_true)), 'float32'))

    def result(self):
        accuracy = self.correct_predictions / (self.total_predictions + K.epsilon())
        return accuracy
