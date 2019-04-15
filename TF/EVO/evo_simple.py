# -*- coding: utf-8 -*-
#author: easo
import numpy as np
import matplotlib.pyplot as plt 
DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200      # 代数 可以理解为循环的次数
X_BOUND = [0, 5]         # x upper and lower bounds


def F(x): 
    #得到Y值
    return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): 
    return pred + 1e-3 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): 
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum()) #p为概率
    return pop[idx]  #返回的是选好的种群 shape:（pop_size,dna_size）


def crossover(parent, pop):     # mating process (genes crossover)
    #交叉
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    #变异
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child



if __name__ == '__main__':
    
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA，pop：种群

    plt.ion()       # something about plotting
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, F(x))

    for _ in range(N_GENERATIONS):
        # if _  >= 1 :
        #     if pop_copy.any() == pop.any():
        #         print("True")
        #     else:
        #         print("False!!!!")
        F_values = F(translateDNA(pop))    # compute function value by extracting DNA
        # print(F_values)
        # something about plotting
        if 'sca'  in globals():
            sca.remove() #sca：可视化散点图
        sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5);
        plt.pause(0.05)

        # GA part (evolution)
        fitness = get_fitness(F_values)
        print("Most fitted DNA: ", pop[np.argmax(fitness), :])
        pop = select(pop, fitness)    #按照适应度选择种群
        pop_copy = pop.copy()    #拷贝是因为pop会改变，需要用之前的copy对象进行交叉操作
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child       # parent is replaced by its child

    plt.ioff()
    plt.show()