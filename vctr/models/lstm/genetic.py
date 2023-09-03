import math
import random

import numpy as np
import torch
import vectorbtpro as vbt
from deap import base, creator, tools
from sklearn.metrics import f1_score
from vctr.models.lstm.actions import save_model
from vctr.models.lstm.data import get_train_and_val_loaders_with_close
from vctr.models.lstm.defaults import SEQUENCE_LENGTH

import numpy as np
from sklearn.metrics import f1_score


mps = torch.device('mps')


# Encoding and decoding genomes
def weights_to_genome(net):
    genome = []
    for param in net.parameters():
        genome.extend(param.detach().numpy().flatten())
    return genome


def genome_to_weights(net, genome):
    weights = []
    genome_index = 0
    for param in net.parameters():
        weight_shape = param.shape
        weight_size = torch.tensor(weight_shape).prod().item()
        weight = torch.tensor(genome[genome_index : genome_index + weight_size]).view(weight_shape)
        weights.append(weight)
        genome_index += weight_size
    return weights


def set_weights(net, genome):
    weights = genome_to_weights(net, genome)
    for weight_tensor, weight in zip(net.parameters(), weights):
        weight_tensor.data.copy_(weight)


# Evaluating fitness
def evaluate_fitness(genome, net, data_loader, criterion):
    set_weights(net, genome)

    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target, close_values in data_loader:
            data, target = data.to(mps), target.to(mps)
            output = net(data)
            loss = criterion(output, target, close_values)
            total_loss += loss.item()

    fitness = -total_loss
    return (fitness,)


def get_total_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params


def calculate_compound_score(scores, min_values, max_values, weights):
    """
    Calculates a compound score based on a list of scores, their respective
    minimum and maximum values, and their weights. Scales the scores to a
    0-1 range before applying the weights. Ensures that the final compound score is within the 0-1 range.

    Parameters:
    scores (list): A list of scores.
    min_values (list): A list of minimum values for each score.
    max_values (list): A list of maximum values for each score.
    weights (list): A list of weights for each score.

    Returns:
    float: The compound score, scaled between 0 and 1.
    """
    # Replace NaN values with the minimum value for that score.
    scores = [min_value if math.isnan(score) else score for score, min_value in zip(scores, min_values)]

    # Clip scores that exceed the min/max range.
    scores = [
        min(max(score, min_value), max_value)
        for score, min_value, max_value in zip(scores, min_values, max_values)
    ]

    # Scale each score to a 0-1 range based on its min/max values.
    scaled_scores = [
        (score - min_value) / (max_value - min_value)
        for score, min_value, max_value in zip(scores, min_values, max_values)
    ]

    # Ensure weights are normalized (i.e., they sum up to 1)
    weights = [weight / sum(weights) for weight in weights]

    # Multiply each scaled score by its weight.
    weighted_scores = [scaled_score * weight for scaled_score, weight in zip(scaled_scores, weights)]

    # Sum up all the weighted scores.
    compound_score = sum(weighted_scores)

    # Ensure the compound score is in the 0-1 range
    assert 0 <= compound_score <= 1, 'The compound score is outside the 0-1 range!'

    # Log the scores before weights are applied
    # print('Scores (before weights):', scores)

    return compound_score


def custom_f1_multiclass(y_true, y_pred):
    # convert to numpy arrays if they are not
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # initialize counts for each class
    true_positives = [0, 0, 0]
    false_positives = [0, 0, 0]
    false_negatives = [0, 0, 0]

    # iterate over the arrays
    for i in range(len(y_true)):
        # handle the "off by one" rule for minority classes
        if y_pred[i] in [1, 2] and y_true[i] == 0:
            if (i > 0 and y_true[i - 1] == y_pred[i]) or (i < len(y_true) - 1 and y_true[i + 1] == y_pred[i]):
                true_positives[y_pred[i]] += 1
            else:
                false_positives[y_pred[i]] += 1
                false_negatives[y_true[i]] += 1
        elif y_true[i] == y_pred[i]:
            true_positives[y_true[i]] += 1
        else:
            false_positives[y_pred[i]] += 1
            false_negatives[y_true[i]] += 1

    # calculate precision, recall, and F1 for each class
    precisions = [tp / (tp + fp) if tp + fp > 0 else 0 for tp, fp in zip(true_positives, false_positives)]
    recalls = [tp / (tp + fn) if tp + fn > 0 else 0 for tp, fn in zip(true_positives, false_negatives)]
    f1_scores = [2 * (p * r) / (p + r) if p + r > 0 else 0 for p, r in zip(precisions, recalls)]

    # return the average F1 score
    return sum(f1_scores) / len(f1_scores)


def run_genetic_algorithm(
    model,
    symbol=None,
    timeframes=None,
    start=None,
    end=None,
    test_pct=0.25,
    label_args=(0.065, 0.005),
    batch_size=256,
    ngen=100,
    cxpb=0.5,
    population_size=50,
    mutpb=0.2,
    sequence_length=SEQUENCE_LENGTH,
    device=mps,
    generational_cb=None,
):
    def custom_profit_loss(logits, target, close_values):
        freq = timeframes[0].replace('m', 'T').replace('h', 'H').replace('d', 'D')

        _, preds = torch.max(logits, dim=1)

        # Run a vectorbtpro backtest to get the profit percentage.
        p = preds.cpu().numpy()
        pf = vbt.Portfolio.from_signals(
            close_values,
            p == 1,
            p == 2,
            freq=freq,
            init_cash=1,
            fees=0.0006,
            slippage=0.001,
            log=True,
        )

        f1 = custom_f1_multiclass(target.cpu().numpy(), preds.cpu().numpy())

        scores = [
            pf.trades.profit_factor,
            pf.sortino_ratio,
            pf.trades.win_rate,
            f1,
            pf.get_total_return(),
        ]

        sortino_weight = 1
        f1_weight = 2
        pf_weight = 1
        win_rate_weight = 5
        returns_weight = 3

        fitness_score = calculate_compound_score(
            scores,
            [0, -20, 0, 0, -1],
            [100, 100, 1, 1, 100000],
            [pf_weight, sortino_weight, win_rate_weight, f1_weight, returns_weight],
        )

        # print(f'==>> fitness_score: {fitness_score}')

        return torch.tensor(-fitness_score, dtype=torch.float32)

    train_loader, val_loader = get_train_and_val_loaders_with_close(
        end=end,
        start=start,
        symbol=symbol,
        timeframes=timeframes,
        lookback=sequence_length,
        label_args=label_args,
        batch_size=batch_size,
        test_pct=test_pct,
    )

    genome_size = get_total_parameters(model)

    # Genetic Algorithm components
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('attr_float', np.random.uniform, -1, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=genome_size)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register('select', tools.selTournament, tournsize=10)
    toolbox.register(
        'evaluate', evaluate_fitness, net=model, data_loader=train_loader, criterion=custom_profit_loss
    )

    # Create the initial population
    pop = toolbox.population(n=population_size)
    best_fitness = -1e9

    elites = tools.selBest(pop, int(0.05 * len(pop)))

    def calculate_fitness(individuals):
        # Ensure every individual in the population has a valid fitness value
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

    for gen in range(ngen):
        calculate_fitness(pop)

        # Select the elites from the original population
        elites = tools.selBest(pop, int(0.05 * len(pop)))

        # Select and clone the parents from the population
        parents = toolbox.select(pop, len(pop))
        parents = list(map(toolbox.clone, parents))

        # Crossover
        for child1, child2 in zip(parents[::2], parents[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in parents:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        calculate_fitness(parents)

        # Elitism: keep the elites in the new population
        pop[: len(elites)] = elites

        # Fill the rest of the new population with the best individuals from the offspring
        pop[len(elites) :] = tools.selBest(parents, len(pop) - len(elites))

        best_ind = tools.selBest(pop, 1)[0]
        # Print fitness scores at the end of each generation
        print(f'Generation: {gen+1}, Best fitness: {best_ind.fitness.values[0]}')

        if best_ind.fitness.values[0] > best_fitness:
            best_fitness = best_ind.fitness.values[0]
            best_genome = best_ind

        if generational_cb is not None:
            generational_cb(gen, best_ind.fitness.values[0])

        save_model(model, model.name)
        save_model(model, 'latest')

    # Get the best individual and set the neural network weights
    best_genome = tools.selBest(pop, 1)[0]
    set_weights(model, best_genome)

    # Evaluate the trained model on a separate validation or test dataset
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, target, close_values in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = custom_profit_loss(output, target, close_values)
            total_loss += loss.item()

    print(f'Validation loss: {total_loss / len(val_loader)}')

    return best_fitness
