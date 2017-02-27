import math
import copy

#FISH_NAMES = ['Salmon','Bass','Trout','Catfish','Tuna','Shark','Eel','Flounder']
GOAL_VAL = "Fish Species"
INDEX_VAL = "Name"
def makeTreeID3(data, attributes, goalAttr):
	vals = []
	attrIndex = attributes.index(goalAttr)
	for d in data:
		vals.append(d[attrIndex])
	if uniform(vals):
		return set(vals).pop()
	counts = {}
	for v in vals:
		if v not in counts.keys():
			counts[v] = 1
		else:
			counts[v] = counts[v] + 1
	largest = ""
	largestCount=0
	for key,value in counts.items():
		if value>largestCount:
			largest=key
			largestCount=value
	if not data or len(attributes)<=1:
		return largest
	else:
		best_attr = bestAttr(attributes, data, attrIndex)
		bestIndex = attributes.index(best_attr)
		tree = {best_attr:{}}
		attrValues = []
		for d in data:
			attrValues.append(d[bestIndex])
		attrValues = list(set(attrValues))
		for v in attrValues:
			newData = []
			for d in data:
				if d[bestIndex]==v:
					newD = copy.deepcopy(d)
					del newD[bestIndex]
					newData.append(newD)
			attrCopy = copy.deepcopy(attributes)
			attributes.remove(best_attr)
			
			newAttributes = copy.deepcopy(attributes)
			newtree = makeTreeID3(newData, newAttributes, goalAttr)
			tree[best_attr][v] = newtree
			attributes = copy.deepcopy(attrCopy)
	return tree

def bestAttr(attributes, data, attrIndex):
	maxGain = 0
	maxGainAttr = ""
	for currAttr in list(set(attributes)-set([GOAL_VAL,INDEX_VAL])):
		currAttrIndex = attributes.index(currAttr)
		currGain = getGain(currAttrIndex, data, attrIndex)
		if currGain > maxGain:
			maxGain = currGain
			maxGainAttr = currAttr
	return maxGainAttr

def getGain(currAttrIndex, data, attrIndex):
	currentEntropy = getEntropy(data, attrIndex)
	newEntropy=0
	counts = {}
	for d in data:
		if counts.has_key(d[currAttrIndex]):
			counts[d[currAttrIndex]]= counts[d[currAttrIndex]] +1
		else:
			counts[d[currAttrIndex]]=1
	for c in counts.keys():
		probability = float(counts[c])/float(sum(counts.values()))
		newData=[]
		for row in data:
			if row[currAttrIndex] == c:
				newData.append(row)
		newEntropy = newEntropy + (probability * getEntropy(newData, attrIndex))
	return currentEntropy-newEntropy

def getEntropy(data, attrIndex):
	entropy = 0.0
	counts = {}
	for d in data:
		if counts.has_key(d[attrIndex]):
			counts[d[attrIndex]]= counts[d[attrIndex]] +1
		else:
			counts[d[attrIndex]]=1
	totalRows = len(data)
	for c in counts.keys():
		probability = float(counts[c])/float(totalRows)
		entropy = entropy + (-1)*(probability)*math.log(probability,2)
	return entropy

def uniform(vals):
	return len(set(vals))==1
