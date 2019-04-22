import pandas as pd
import os 

# indexList = os.listdir('./model/model')
# ckpfile = [os.listdir('./model/model/'+i) for i in indexList]
# layerNums = []
# accList = []
# paraNums = []

# for i,ckp in enumerate(ckpfile):
#     fname = ckp[0]
#     sp = fname.split('-')
#     layerNums.append(int(i/10)+1)
#     accList.append(float(sp[2]))
#     paraNums.append((float(sp[-1].split('.')[0])))

# paraPerlayer = list(map(lambda x,y: x/y,paraNums,layerNums))

# dfDict = {'layernums':layerNums,
#           'acc':accList,
#           'paraNums':paraNums,
#           'paraPerlayer':paraPerlayer
# }

# initaildf = pd.DataFrame(dfDict)
# initaildf.to_csv('./initail.csv')


if __name__ == "__main__":
    initailDf = pd.read_csv('./logs/initail.csv')
    layerNumsAvg = []
    accAvg = []
    accMin = []
    accMax = []
    for nums in range(1,11):
        layerNumsAvg.append(nums)
        tmp = initailDf[initailDf['layernums'] == nums]
        accAvg.append(sum(tmp['acc'])/len(tmp['acc']))
        accMin.append(min(tmp['acc']))
        accMax.append(max(tmp['acc']))
    dictTmp = { 'layerNums':layerNumsAvg,
                'accAvg':accAvg,
                'accMin':accMin,
                'accMax':accMax
    }
    df = pd.DataFrame(dictTmp)
    df.to_csv('./logs/avg.csv')

    