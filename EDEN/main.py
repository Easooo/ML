#author:Easo
#date:2019年4月30日

import Ga
from networks import torch
import train
import os
from dataSet import allTransformList
def createDir(popSize,rootDir='./model/'):
    '''
    创建每个个体的子目录
    '''
    for i in range(popSize):
        os.mkdir(rootDir+str(i))

def evaluate(pop,inputSize,inputChannel,epoch,gen,tfList,dataType):
    '''
    计算种群
    '''
    for popMember in pop:
        popNet = train.trainStep(popMember,popMember[6],inputSize,inputChannel,epoch,tfList,dataType)
        fitness = Ga.getFitness(popMember)
        saveDir = './model/' + str(popMember[6])
        saveName = str(gen)+'-'+str(epoch)+'-'+str(popMember[3][0])+'-'+str(popMember[3][1])+'-'+str(popMember[4])+'.pth'
        torch.save(popNet.state_dict(), saveDir+'/'+saveName)

if __name__ == "__main__":
    GaForLab = Ga.Genetic()

    generation = 10
    epoch = 3
    populationSize = 100

    print('input dataType:')
    dataType = input()

    print('inputsize:')
    inputSize = input()
    inputSize = int(inputSize)

    print('inputchannel:')
    inputChannel = input()
    inputChannel = int(inputChannel)

    tfList = allTransformList(dataType)

    population = GaForLab.createPopulation(populationSize)  #初始化种群
    createDir(100)   #创建文件夹
    # evaluate(population,inputSize=28,inputChannel=1,epoch=epoch,gen=10000)  #test
    train.savePopCsv('./logs/'+'initalpop.csv',population)
    for gen in range(generation):

        if gen !=0 :
            epoch += 1
            populationSize = populationSize - 10

        print('===================gen:%d===================='%(gen))
        evaluate(population,inputSize,inputChannel,epoch,gen,tfList=tfList,dataType=dataType)
        populationTmp = population.copy()
        populationTmp = sorted(populationTmp,key=lambda popmember:popmember[4]) #排序
        population = populationTmp[:populationSize]

        parent,pidx = GaForLab.selcet(population)
        offspring1 = parent.copy()
        GaForLab.mutate(offspring1)
        offspring2 = offspring1.copy()
        GaForLab.mutate(offspring2)
        print('============parent fit:%f================='%(parent[4]))

        print('====================mutate1=======================')
        offspringNet1 = train.trainStep(offspring1,offspring1[6],inputSize,inputChannel,epoch,tfList,dataType)
        offspringFit1 = Ga.getFitness(offspring1)
        print('============off1 fit:%f================='%(offspring1[4]))

        print('====================mutate2=======================')
        offspringNet2 = train.trainStep(offspring2,offspring2[6],inputSize,inputChannel,epoch,tfList,dataType)
        offspringFit2 = Ga.getFitness(offspring2)
        print('============off2 fit:%f================='%(offspring2[4]))
        mutateList = sorted([parent,offspring1,offspring2],key=lambda popmember:popmember[4])
        best = mutateList[0]
        population[pidx] = best

        if best == offspring1:
            print('================return offspring1================')
            print('=========popindex:%d=========='%(best[6]))
            saveDir = './model/' + str(offspring1[6])
            saveName = 'ckp-'+str(epoch)+'-'+str(offspring1[3][0])+'-'+str(offspring1[3][1])+'-'+'offspring1'+'.pth'
            torch.save(offspringNet1.state_dict(), saveDir+'/'+saveName)

        elif best == offspring2:
            print('================return offspring2================')
            print('=========popindex:%d=========='%(best[6]))
            saveDir = './model/' + str(offspring2[6])
            saveName = 'ckp-'+str(epoch)+'-'+str(offspring2[3][0])+'-'+str(offspring2[3][1])+'-'+'offspring2'+'.pth'
            torch.save(offspringNet2.state_dict(), saveDir+'/'+saveName)

        os.mkdir('./logs/'+str(gen))
        train.savePopCsv('./logs/'+str(gen)+'/'+'pop.csv',population)
        