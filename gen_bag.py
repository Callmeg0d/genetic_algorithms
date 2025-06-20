import random
import time

with open("benchmarks/knapsack/p7/cap.txt", "r") as f:
    capacity = int(f.readline().strip())

profits = []
with open("benchmarks/knapsack/p7/prof.txt", "r") as f:
    for line in f:
        profits.append(int(line.strip()))

weights = []
with open("benchmarks/knapsack/p7/weig.txt", "r") as f:
    for line in f:
        weights.append(int(line.strip()))


def initialize_population(num_items, population_size):
    return [[random.randint(0, 1) for _ in range(num_items)]
            for _ in range(population_size)]


def fitness(individual, weights, profits, capacity):
    total_weight = sum(w * i for w, i in zip(weights, individual)) #суммарный вес
    if total_weight > capacity:
        return 0
    return sum(v * i for v, i in zip(profits, individual)) #суммарная ценность


def tournament_selection(population, weights, profits, capacity, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: fitness(ind, weights, profits, capacity)) #берём max по fitness


def k_point_crossover(parent1, parent2, crossover_points):
    length = len(parent1)
    points = sorted(random.sample(range(1, length), crossover_points)) #выбираем случайные точки для деления на отрезки

    child1, child2 = parent1.copy(), parent2.copy()

    for i in range(len(points) + 1): #обработка каждого отрезка
        start = 0 if i == 0 else points[i - 1]
        end = length if i == len(points) else points[i]

        if i % 2 != 0: # обмен в нечётных отрезках
            child1[start:end], child2[start:end] = child2[start:end], child1[start:end]

    return child1, child2


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate: #делаем рандом мутацию (замена на противоположный)
            individual[i] = 1 - individual[i]
    return individual


def remove_random_items(individual, weights, profits, capacity):
    total_weight = sum(w * i for w, i in zip(weights, individual))

    while total_weight > capacity:
        selected_indices = [i for i, val in enumerate(individual) if val == 1]

        if not selected_indices:
            return individual

        #выбираем случайный предмет для удаления
        item_to_remove = random.choice(selected_indices)
        individual[item_to_remove] = 0
        total_weight -= weights[item_to_remove]

    return individual


def gen_result(weights, profits, capacity,
               population_size, generations,
               tournament_size, crossover_points,
               mutation_rate):
    num_items = len(weights)
    population = initialize_population(num_items, population_size)
    best_individual = None
    best_fitness = 0

    start_time = time.time()

    for generation in range(generations):
        #Делаем проверку на вместимость
        population = [remove_random_items(ind, weights, profits, capacity) for ind in population]

        fitness_scores = [fitness(ind, weights, profits, capacity) for ind in population]

        current_best = max(fitness_scores)
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[fitness_scores.index(current_best)].copy()

        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, weights, profits, capacity, tournament_size)
            parent2 = tournament_selection(population, weights, profits, capacity, tournament_size)

            child1, child2 = k_point_crossover(parent1, parent2, crossover_points)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            #Делаем проверку на вместимость перед добавлением
            child1 = remove_random_items(child1, weights, profits, capacity)
            child2 = remove_random_items(child2, weights, profits, capacity)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    execution_time = time.time() - start_time

    return best_individual, best_fitness, execution_time


population_size = 150
generations = 200
tournament_size = 4
crossover_points = 2
mutation_rate = 0.11


best_solution, best_value, exec_time = gen_result(
    weights, profits, capacity, population_size, generations,
    tournament_size, crossover_points, mutation_rate
)

print("Генетический алгоритм")
print(f"Время выполнения: {exec_time:.6f} секунд")
print("Лучшее решение:")
for item in best_solution:
    print(item)
print(f"Общий вес: {sum(w * i for w, i in zip(weights, best_solution))}")
print(f"Значение целевой функции: {best_value}")