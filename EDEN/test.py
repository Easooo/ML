from Ga import Genetic,departLayers,mergeLayers,judgeConv,getFitness
from main import *
from train import *

if __name__ == "__main__":
    
    test = Genetic()
    testPop = test.createPopulation(100)
    testPoptmp = testPop.copy()
    for i in range(100):
        aa = testPop[i]
        test.mutate(aa)
        # dd = departLayers(aa[1])
        # ee = mergeLayers(dd)
        # if aa[1] == ee:
        #     print('right')
        # print(aa[0],len(dd)-1)
        # print(aa[0] == len(dd)-1)
        # print(judgeConv(dd))
    createDir(100)
    evaluate(testPop,28,1,1,1)

