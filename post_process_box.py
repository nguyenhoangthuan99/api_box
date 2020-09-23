import cv2
import numpy as np
import math

# convert file txt to list box


def getBoxTxt(boxes):
    
    
    listBox = []
    for box in boxes:
        
        listBox.append([[box[0], box[1]], [box[2], box[3]],
                        [box[6], box[7]], [box[4], box[5]]])
    return listBox

# distance between 2 points


def distancePoint(pointA, pointB):
    return math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)

# distance between points and line, pointA is point , pointB-pointC is line


def distancePointLine(pointA, pointB, pointC):
    ab = distancePoint(pointA, pointB)
    ac = distancePoint(pointA, pointC)
    cb = distancePoint(pointC, pointB)
    acb = math.acos((ac**2 + cb**2 - ab**2) / (2 * ac * cb))
    _range = ac * math.sin(acb)
    return _range

# Box
"""
    X1(x,y)---------Y1
    -               -
    -               -
    -               -
    X0--------------Y0

"""


class Box(object):

    def __init__(self, box):
        super(Box, self).__init__()
        self.box = box

    def getX0(self):
        return [self.box[0][0], self.box[0][1]]

    def getY0(self):
        return [self.box[1][0], self.box[1][1]]

    def getX1(self):
        return [self.box[2][0], self.box[2][1]]

    def getY1(self):
        return [self.box[3][0], self.box[3][1]]


# if listB = [[1,2],[2,3],[5,6]] -> [[1,2,3],[5,6]], boxs in line
def dupli(listB):
    listD = [listB[0]]
    for i in listB:
        cout = 0
        for j in listD:
            if(i[0] in j and i[1] not in j):
                j.append(i[1])
                cout = 1
            elif(i[1]in j and i[0] not in j):
                j.append(i[0])
                cout = 1
            elif(i[1]in j and i[0] in j):
                cout = 1
        if(cout == 0):
            listD.append(i)
    return listD


# box not in line, it is alone
def checkBoxAlone(listA, listB):
    listC = []
    for i in listA:
        cout = 0
        for j in listB:
            if(i not in j):
                cout += 1
        if(cout == len(listB)):
            listC.append(i)
    return listC

# save box to bmp, extendA is height, extendB is width


def saveBox(pathFile, box, index, extendA, extendB):
    image = pathFile #cv2.imread(pathFile.split(".")[0] + ".jpg")
    a1 = box.getY1()[0] - box.getX1()[0]
    a0 = box.getY0()[0] - box.getX0()[0]
    b1 = box.getX1()[1] - box.getX0()[1]
    b0 = box.getY1()[1] - box.getY0()[1]
    pt1 = np.float32([[box.getX0()[0] - extendA * a0, box.getX0()[1] - extendB * b1], [box.getY0()[0] + extendA * a0, box.getY0()[1] - extendB * b0], [
                     box.getX1()[0] - extendA * a1, box.getX1()[1] + extendB * b1], [box.getY1()[0] + extendA * a1, box.getY1()[1] + extendB * b0]])
    pt2 = np.float32([[0, 0], [128, 0], [0, 32], [128, 32]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    result = cv2.warpPerspective(image, matrix, (128, 32))
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("license/"+pathFile.split("/")[-1][:-4] + "_" + index + ".bmp", result)
    return (result,pt1)
# merge boxs in line, (x0,x1) top left, (x1,y1) top right


def merge(boxList):

    # x1x = min([box.getX1()[0] for box in boxList])
    # x1y = max([box.getX1()[1] for box in boxList])
    # y1x = max([box.getY1()[0] for box in boxList])
    # y1y = max([box.getY1()[1] for box in boxList])
    x0x = 100000
    left = 0
    y1x = 0
    right = 0
    for i in range(len(boxList)):
        if (x0x > boxList[i].getX0()[0]):
            x0x = boxList[i].getX0()[0]
            left = i
        if(y1x < boxList[i].getY1()[0]):
            y1x = boxList[i].getY1()[0]
            right = i
    # x0y = min([box.getX0()[1] for box in boxList])
    # y0x = max([box.getY0()[0] for box in boxList])
    # y0y = min([box.getY0()[1] for box in boxList])
    # return Box([[x0x, x0y], [y0x, y0y], [x1x, x1y], [y1x, y1y]])
    return Box([boxList[left].getX0(), boxList[right].getY0(), boxList[left].getX1(), boxList[right].getY1()])


def logic(boxes,image):
    boxList = []
    boxOnly = []
    listBox = [Box(box) for box in getBoxTxt(boxes)]
    print(len(listBox))
    # saveBox(pathFile, listBox[1], "Test2", 0.01, 0.05)
    # if number box = 1 -> save
    result = []
    bbb = []
    if (len(listBox) == 1):
        result = [saveBox(image, listBox[0], "", 0.00, 0.00)[0]]
        bbb = [saveBox(image, listBox[0], "", 0.00, 0.00)[1]]
    for indexI in range(len(listBox) - 1):
        for indexJ in range(indexI + 1, len(listBox)):
           
            y = max(distancePointLine(listBox[indexI].getX1(), listBox[indexJ].getX0(), listBox[indexJ].getY0(
            )), distancePointLine(listBox[indexJ].getX1(), listBox[indexI].getX0(), listBox[indexI].getY0()))
            x = max(distancePointLine(listBox[indexI].getX1(), listBox[indexJ].getY1(), listBox[indexJ].getY0(
            )), distancePointLine(listBox[indexJ].getX1(), listBox[indexI].getY1(), listBox[indexI].getY0()))
        
            yCheck = distancePoint(listBox[indexI].getX1(), listBox[indexI].getX0(
            )) + distancePoint(listBox[indexJ].getX1(), listBox[indexJ].getX0())
            xCheck = distancePoint(listBox[indexI].getY1(), listBox[indexI].getX1(
            )) + distancePoint(listBox[indexJ].getY1(), listBox[indexJ].getX1())
       
            if(y > 1.2 * yCheck or x > 1.2 * xCheck):
                i = 1  # away
            else:
                print(yCheck, y)
                if(0.7 * yCheck > y):
                    boxList.append([indexJ, indexI])  # in the same line
                else:
                    boxOnly.append(indexI)  # alone
                    boxOnly.append(indexJ)

    boxOnly = list(set(boxOnly))
    if (len(boxList) != 0):
        boxList = dupli(boxList)  # in the same line
        if (len(boxOnly) != 0):
            boxOnly = checkBoxAlone(boxOnly, boxList)  # check box alone
            if len(boxOnly) != 0:
                cout = 0
                for box in boxOnly:
                    cout += 1
                    img , pt = saveBox(image, listBox[box], str(cout) + "y", 0.00, 0.00)
                    result.append(img)
                    bbb.append(pt)
        cout = 0
        for boxLine in boxList:  # save box in the same line
            cout += 1
            boxLine = [listBox[index] for index in boxLine]
            img , pt = saveBox(image, merge(boxLine), str(cout) + "x", 0.00, 0.00)
            result.append(img)
            bbb.append(pt)

    else:  # if only box alone
        if (len(boxOnly) != 0):
            cout = 0
            for box in boxOnly:  # save boxs alone
                cout += 1
                img , pt = saveBox(image, listBox[box], str(cout) + "y", 0.00, 0.00)
                result.append(img)
                bbb.append(pt)

    #print(boxList)
   # print(boxOnly)
    return result,bbb
"""
import glob

file1 = glob.glob("annos/*.txt")
for i in file1:
    logic(i)

"""

#logic("34.txt")
