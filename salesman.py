import random
import math
import time
import numpy as np


def read_tsp_file(filename):
    cities = []
    dist_matrix = None
    edge_weight_type = None
    edge_weight_format = None
    dimension = 0
    weights = []
    reading_weights = False
    reading_coords = False

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            edge_weight_format = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line == "EDGE_WEIGHT_SECTION":
            reading_weights = True
            reading_coords = False
            continue
        elif line == "NODE_COORD_SECTION":
            reading_coords = True
            reading_weights = False
            continue
        elif line == "EOF":
            break

        elif reading_weights:
            if line in ["DISPLAY_DATA_SECTION", "NODE_COORD_SECTION", "EOF"]:
                reading_weights = False
                continue
            weights.extend(map(int, line.split()))
        elif reading_coords and line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 3:
                x, y = map(float, parts[1:3])
                cities.append((x, y))

    # обработка матрицы
    if edge_weight_type == "EXPLICIT":
        dist_matrix = np.zeros((dimension, dimension), dtype=int)
        if edge_weight_format == "FULL_MATRIX":
            idx = 0
            for i in range(dimension):
                for j in range(dimension):
                    dist_matrix[i][j] = weights[idx]
                    idx += 1
        elif edge_weight_format == "LOWER_DIAG_ROW":
            idx = 0
            for i in range(dimension):
                for j in range(i + 1):
                    dist_matrix[i][j] = weights[idx]
                    dist_matrix[j][i] = weights[idx]
                    idx += 1
        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")

        return None, dist_matrix, edge_weight_type

    else:
        return cities, None, edge_weight_type


def euclidean_distance(city1, city2):
    xd = city1[0] - city2[0]
    yd = city1[1] - city2[1]
    return int(round(math.sqrt(xd * xd + yd * yd)))

def att_distance(city1, city2):
    xd = city1[0] - city2[0]
    yd = city1[1] - city2[1]
    rij = math.sqrt((xd**2 + yd**2) / 10.0)
    tij = int(rij + 0.5)
    return tij + 1 if tij < rij else tij

def precompute_distances(cities, metric):
    n = len(cities)
    dist_matrix = np.zeros((n, n), dtype=int)

    if metric == 'EUC_2D':
        distance_fn = euclidean_distance
    elif metric == 'ATT':
        distance_fn = att_distance

    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_fn(cities[i], cities[j])
            dist_matrix[i][j] = dist_matrix[j][i] = dist

    return dist_matrix


def route_length_fast(route, dist_matrix):
    return sum(dist_matrix[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))


def initialize_population(num_cities, population_size):
    return [random.sample(range(num_cities), num_cities) for _ in range(population_size)]


def tournament_selection(population, dist_matrix, tournament_size):
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: route_length_fast(x, dist_matrix))


def pmx_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    def create_child(p1, p2):
        child = [-1] * size
        child[start:end] = p1[start:end]
        mapping = {p1[i]: p2[i] for i in range(start, end)}

        for i in range(size):
            if child[i] == -1:
                candidate = p2[i]
                while candidate in child:
                    candidate = mapping.get(candidate, candidate)
                child[i] = candidate
        return child

    return create_child(parent1, parent2), create_child(parent2, parent1)


def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(individual)), 2))
        individual[i:j + 1] = reversed(individual[i:j + 1])
    return individual


def genetic_algorithm(dist_matrix, population_size, generations,
                          tournament_size, mutation_rate, elite_size):
    num_cities = len(dist_matrix)
    population = initialize_population(num_cities, population_size)
    best_individual, best_length = None, float('inf')
    start_time = time.time()

    print(f"Params: pop_size={population_size}, gens={generations}")
    print("=" * 60)

    for generation in range(generations):
        population.sort(key=lambda x: route_length_fast(x, dist_matrix))
        current_length = route_length_fast(population[0], dist_matrix)

        if current_length < best_length:
            best_individual, best_length = population[0].copy(), current_length
            print(f"Gen {generation:4d}: New best = {best_length}")

        new_population = population[:elite_size]
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, dist_matrix, tournament_size)
            parent2 = tournament_selection(population, dist_matrix, tournament_size)
            child1, child2 = pmx_crossover(parent1, parent2)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    exec_time = time.time() - start_time
    print("=" * 50)
    print(f"Итоговое значение: {best_length}")
    print(f"Время выполнения: {exec_time:.2f} sec")
    return best_individual, best_length


if __name__ == "__main__":
    filename = "benchmarks/traveling salesman/a280.tsp"
    cities, dist_matrix, metric = read_tsp_file(filename)

    if dist_matrix is None:
        dist_matrix = precompute_distances(cities, metric)
    # Гиперпараметры
    best_route, best_length = genetic_algorithm(
        dist_matrix=dist_matrix,
        population_size=1500,
        generations=2100,
        tournament_size=7,
        mutation_rate=0.09,
        elite_size=40
    )

    print(f"Итоговый путь: {best_route}")
