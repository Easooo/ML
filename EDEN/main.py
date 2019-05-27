#author:Easo
#date:2019年4月30日

import Ga
from networks import torch
import train
import os
from dataSet import allTransformList
import copy

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
        oldModeldir = None
        saveDir = './model/' + str(popMember[7])
        fileList = os.listdir(saveDir)

        if len(fileList) != 0:
            for i in fileList:
                if(os.path.splitext(i)[1]) == '.pth':
                    oldModeldir = saveDir + '/' + i

        print('======old model:%s======'%(oldModeldir)) 
        popNet = train.trainStep(popMember,inputSize,inputChannel,epoch,tfList,dataType,oldModeldir)
        fitness = Ga.getFitness(popMember)
        if oldModeldir is not None:
            os.remove(oldModeldir)
        saveName = str(gen)+'-'+str(popMember[3])+'-'+\
                str(popMember[4])+'-'+str(popMember[5])+'.pth'
        with open(saveDir+'/'+'modellog.txt','a+') as f:
            f.write(saveName+'\n')
        torch.save(popNet.state_dict(), saveDir+'/'+saveName)

if __name__ == "__main__":
    GaForLab = Ga.Genetic()

    generation = 10
    epoch = 3
    populationSize = 100

    dataType = input('input dataType:')

    inputSize = input('inputsize:')
    inputSize = int(inputSize)

    inputChannel = input('inputchannel:')
    inputChannel = int(inputChannel)

    tfList = allTransformList(dataType)

    population = GaForLab.createPopulation(populationSize)  #初始化种群
    createDir(populationSize)   #创建文件夹
    # evaluate(population,inputSize=28,inputChannel=1,epoch=epoch,gen=10000)  #test
    train.savePopCsv('./logs/'+'initalpop.csv',population)
    for gen in range(generation):
        print('===================gen:%d===================='%(gen))
        if gen !=0 :
            epoch += 1
            populationSize = populationSize - 10
            evaluate(population,inputSize,inputChannel,1,gen,tfList=tfList,dataType=dataType)
        else:
            evaluate(population,inputSize,inputChannel,epoch,gen,tfList=tfList,dataType=dataType)
        populationTmp = copy.deepcopy(population)
        populationTmp = sorted(populationTmp,key=lambda popmember:popmember[5]) #排序
        population = populationTmp[:populationSize]


        ##############变异子代####################
        mutateTimes = int(100/populationSize)
        for m in range(mutateTimes):
            
            parent,pidx = GaForLab.selcet(population)
            offspring1 = copy.deepcopy(parent)
            GaForLab.mutate(offspring1)
            offspring2 = copy.deepcopy(offspring1)
            GaForLab.mutate(offspring2)
            print('============parent idx:%d================='%(parent[7]))

            print('====================mutate1=======================')
            offspringNet1 = train.trainStep(offspring1,inputSize,inputChannel,epoch,tfList,dataType)
            offspringFit1 = Ga.getFitness(offspring1)
            print('============off1 fit:%f================='%(offspring1[5]))

            print('====================mutate2=======================')
            offspringNet2 = train.trainStep(offspring2,inputSize,inputChannel,epoch,tfList,dataType)
            offspringFit2 = Ga.getFitness(offspring2)
            print('============off2 fit:%f================='%(offspring2[5]))
            mutateList = sorted([parent,offspring1,offspring2],key=lambda popmember:popmember[5])
            best = mutateList[0]
            population[pidx] = best

            if best == offspring1:
                print('================return offspring1================')
                print('=========popindex:%d=========='%(best[7]))
                oldModeldir = None
                saveDir = './model/' + str(offspring1[7])
                fileList = os.listdir(saveDir)

                if len(fileList) != 0:
                    for i in fileList:
                        if(os.path.splitext(i)[1]) == '.pth':
                            oldModeldir = saveDir + '/' + i
                            os.remove(oldModeldir)        
                saveName = 'm-'+str(epoch)+'-'+str(offspring1[3])+'-' \
                + str(offspring1[4])+'-'+str(offspring1[5])+'-offspring1'+'.pth'
                with open(saveDir+'/'+'modellog.txt','a+') as f:
                    f.write(str(gen)+'-'+saveName+'\n')
                torch.save(offspringNet1.state_dict(), saveDir+'/'+saveName)

            elif best == offspring2:
                print('================return offspring2================')
                print('=========popindex:%d=========='%(best[7]))
                oldModeldir = None
                saveDir = './model/' + str(offspring2[7])
                fileList = os.listdir(saveDir)

                if len(fileList) != 0:
                    for i in fileList:
                        if(os.path.splitext(i)[1]) == '.pth':
                            oldModeldir = saveDir + '/' + i
                            os.remove(oldModeldir)  

                saveName = 'm-'+str(epoch)+'-'+str(offspring2[3])+'-' \
                + str(offspring2[4])+'-'+str(offspring2[5])+'-offspring2'+'.pth'
                with open(saveDir+'/'+'modellog.txt','a+') as f:
                    f.write(str(gen)+'-'+saveName+'\n')
                torch.save(offspringNet2.state_dict(), saveDir+'/'+saveName)

            #####################结束变异子代########################

        os.mkdir('./logs/'+str(gen))
        train.savePopCsv('./logs/'+str(gen)+'/'+'pop.csv',population)
        