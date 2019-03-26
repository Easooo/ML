# -*- coding: utf-8 -*-
#author: easo
import numpy as np
import matplotlib.pyplot as plt 

N_CITIES = 30       #相当于DNA的size
CROSS_RATE = 0.1        #交叉的几率
MUTATE_RATE = 0.02      #变异几率
POP_SIZE = 500
N_GENERATIONS = 500

class GA(object):
    def __init__(self,DNA_size,CR_rate,Mu_rate,P_size):
        self.DNA_size = DNA_size
        self.CR_rate = CR_rate
        self.Mu_rate = Mu_rate
        self.P_size = P_size
        
        #pop:shape=(P_size,DNA_size),每一行存储DNA_size个数字,表示从城市到城市的路径
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(P_size)])
    
    def translateDNA(self,DNA,city_pos):
        """
        lx:shape=(P_size,DNA_size) ly:shape=(P_size,DNA_size) 是和pop一样shape的空数组
        这里的DNA和pop一样
        city_pos为城市的位置，shape=(30,2)分别表示30个城市的横坐标和纵坐标
        city_coord为dna中每一个个体城市的位置
        """
        lx = np.empty_like(DNA,dtype=np.float64)
        ly = np.empty_like(DNA,dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_pos[d]
            lx[i, :] = city_coord[:, 0]
            ly[i, :] = city_coord[:, 1]
        return lx, ly
    
    def get_fitness(self,lx,ly):
        """
        total_distance存放pop中每一个个体的城市路径总长度s
        """
        total_distance = np.empty((lx.shape[0],),dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(lx, ly)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.P_size), size=self.P_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        """
        交叉过后不能有重复的城市
        """
        if np.random.rand() < self.CR_rate:
            i_ = np.random.randint(0, self.P_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            keep_city = parent[~cross_points]                                       # ~表示取反
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        """
        直接交换
        """
        for point in range(self.DNA_size):
            if np.random.rand() < self.Mu_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class TravelSalesPerson(object):
    def __init__(self, n_cities):
        """
        city_position:shape=(n_cities,2) n_cities=30
        """
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'b-')
        plt.text(0.1, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 16, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)


#创建类
ga = GA(DNA_size=N_CITIES, CR_rate=CROSS_RATE, Mu_rate=MUTATE_RATE, P_size=POP_SIZE)

env = TravelSalesPerson(N_CITIES)

for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)
    fitness, total_distance = ga.get_fitness(lx, ly)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)

    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

plt.ioff()
plt.show()