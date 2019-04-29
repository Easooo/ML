from Ga import Genetic,departLayers,mergeLayers,judgeConv

if __name__ == "__main__":
    
    test = Genetic()
    testPop = test.createPopulation(100)
    for i in range(100):
        aa = testPop[i]
        dd = departLayers(aa[1])
        ee = mergeLayers(dd)
        # if aa[1] == ee:
        #     print('right')
        # print(aa[0],len(dd)-1)
        # print(aa[0] == len(dd)-1)
        print(judgeConv(dd))
    pass

