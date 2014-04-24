import math

### 
# Parses file line by line and returns a 2D list containing the data set
###
def parseFile( path ):
	#Structure of input file
	##########
	# 0 id -> not relevant for prediction
	# 1 age ->continuous
	# 2 workclass-> categorical
	# 3 fntwgt -> continuous
	# 4 education -> categorical
	# 5 education-num -> continuous
	# 6 marital status -> categorical
	# 7 occupation -> categorical
	# 8 relationsip -> categorical
	# 9 race -> categorical
	# 10 sex -> categorical
	# 11 capital gain -> continuous
	# 12 capital loss -> continuous
	# 13 hour-pre-week -> continuous
	# 14 native-country -> categorical
	# 15 target-variable -> we have to predict this!
	##########

	data = [ line.split(',') for line in file(path) ]

	for i in range(0,len(data)):

		# convert from strings to floats and ints.
		data[i][1] = int(data[i][1])
		data[i][3] = int(data[i][3])
		data[i][5] = int(data[i][5])
		data[i][11] = int(data[i][11])
		data[i][12] = int(data[i][12])
		data[i][13] = int(data[i][13])

		#strip \n fron right side of target variable
		data[i][15] = data[i][15].rstrip('\n')

	return data

def splitList( list, start, end ):
	newList = []
	for i in range( start, end ):
		newList.append(list[i])
	return newList
#####################################################
## Decision tree

### Step 1 is to make a representation for the tree
class decisionnode:
	def __init__( self, col=-1, value=None, results=None, tb=None, fb=None):
		self.col = col
		self.value = value
		self.results = results
		self.tb = tb
		self.fb = fb

### Step 2 training the tree
# 2.1 Dividing data based on criterion
def divideset( rows, column, value ):
	
	split_function = None
	if isinstance( value, int ) or isinstance( value, float ):
		split_function = lambda row:row[column] >= value
	else:
		split_function = lambda row:row[column] == value

	#divid rows into two sets and return
	set1 = [ row for row in rows if split_function(row) ]
	set2 = [ row for row in rows if not split_function(row) ]

	return ( set1, set2 )

# 2.2 choosing best split
def uniquecounts( rows ):
	results = {}
	for row in rows:
		r=row[len(row)-1]
		if r not in results: results[r] = 0
		results[r] += 1
	return results

#use entropy or gini	
def entropy( rows ):
	from math import log
	log2 = lambda x:log(x)/log(2)
	results = uniquecounts(rows)
	#calc ent
	ent = 0.0
	for r in results.keys():
		p = float(results[r])/len(rows)
		ent = ent - p * log2(p)
	return ent

### Step 3 build a tree
def buildTree( rows, scoref=entropy ):
	# 1 find best attribute to split on
	# 1.1 calc ent for whole group
	current_score = scoref(rows)

	best_gain = 0.0
	best_criteria = None
	best_sets = None

	# 1.2 divide up the group
	column_count = len(rows[0]) - 1
	for col in range(0,column_count):
		#1.2.1 generate the list of different values in col
		column_values = {}
		for row in rows:
			column_values[row[col]] = 1
			
		# 1.2.2
		for value in column_values.keys():
			(set1,set2) = divideset( rows, col, value )

			#1.3 calc info gain
			p = float( len(set1)) / len(rows)
			gain = current_score - p * scoref(set1) - (1-p) * scoref(set2)

			#1.4 select attribute with highest info gain
			if gain > best_gain and len(set1) > 0 and len(set2) > 0:
				best_gain = gain
				best_criteria = (col, value )
				best_sets = (set1,set2)

	# 2 create branches
	# 2.1
	if best_gain > 0:
		trueBranch = buildTree(best_sets[0])
		falseBranch = buildTree(best_sets[1])

		return decisionnode( col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
	else:
		return decisionnode(results=uniquecounts(rows))

### Step 4 displaying the tree

### Step 5 classifying New observations
def classify( observation, tree ):
	if tree.results != None:
		return tree.results
	else:
		v = observation[tree.col]
		branch = None
		if isinstance(v,int) or isinstance(v,float):
			if v >= tree.value: branch = tree.tb
			else: branch = tree.fb
		else:
			if v == tree.value: branch=tree.tb
			else: branch=tree.fb
		return classify( observation, branch )

### Step 6 pruning the tree
##needed?

### Step 7 dealing with missing data
##needed?

### Step 8 dealing with Numerical outcomes
##needed?

def convertDecTreeOutput2Label(output):
    maxcount = 0
    maxlabel = ''
    for k in output.keys():
        if output[k]>maxcount:
            maxcount = output[k]
            maxlabel = k
    return maxlabel

## Decision tree End
#####################################################
def main():
	# Step 1 get data
	#queryData = parseFile('queries.txt')
	trainingData = parseFile('trainingset.txt')

	tdOne = splitList( trainingData, 0, 500 )
	tdTwo = splitList( trainingData, 600, 800 )
	# Step 2 train a model
	tree = buildTree(tdOne )
	predictions = []

	for i in range( len(tdTwo) ):
		o = classify( tdTwo[i], tree )
		p = convertDecTreeOutput2Label(o)
		predictions.append(p)

	correct = 0
	for i in range(len(predictions)):
		print("Test query: " + str(tdTwo[i]) + ", Predicted target: " + predictions[i])
		if predictions[i] == tdTwo[i][-1]:
			correct += 1
	print("Percentage correct: " + str("%.4f" % round(float(correct)/len(predictions),4)))

main()