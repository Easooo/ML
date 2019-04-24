from Ga import Genetic

if __name__ == "__main__":
    
    test = Genetic()
    testPop = test.createPopulation(100)
    a = test.selcet(testPop,100)
    pass
