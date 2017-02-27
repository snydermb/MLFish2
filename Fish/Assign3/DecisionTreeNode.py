class DecisionTreeNode:
	value = ""
	next = []
	
	
	def __init__(self, value, nextVals):
	    self.value = value
		if (isinstance(nextVals, dict)):
			self.next = nextVals.keys()
	
	
	def printNode(self):
		print value
		return value
		
	def setVal(self, val):
		self.value = val