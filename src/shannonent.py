import math


def calcShannonEnt(dataSet):
    # numEntries = len(dataSet)
    # labelCounts = {}
    # for featVec in dataSet:
    #     currentLabel = featVec[-1]
    #     if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
    #     labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # for key in labelCounts:
    for prob in dataSet:
        # prob = float(labelCounts[key])/numEntries
        if prob == 0:
            continue
        shannonEnt -= prob * math.log(prob, 2)
    # 2.0
    # 1.3567796494470397
    # 0.6175431233120147
    return shannonEnt


# dataSet = [1.0, 0.0, 0.0, 0.0]
dataSet = [0.25, 0.25, 0.25, 0.25]


def main():
    pass


if __name__ == '__main__':
    main()
    print(calcShannonEnt(dataSet))
























