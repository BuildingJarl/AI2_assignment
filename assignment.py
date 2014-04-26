#Paul Kennedy
#C10319445
#Ivan Bacher
#C10736831

import math

#####################################################
## File Parsing

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

                # convert from strings to floats.
                data[i][1] = float(data[i][1])
                data[i][3] = float(data[i][3])
                data[i][5] = float(data[i][5])
                data[i][11] = float(data[i][11])
                data[i][12] = float(data[i][12])
                data[i][13] = float(data[i][13])

                #strip \n fron right side of target variable
                data[i][15] = data[i][15].rstrip('\n')

        return data

def splitListIntoEqualSize( lst, sz ):

        return [lst[i:i+sz] for i in range(0, len(lst), sz)]

## File Parsing End
#####################################################

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

def gini( rows ):
        total = len( rows )
        counts = uniquecounts( rows )
        imp = 0
        for k1 in counts:
                p1 = float( counts[k1] ) / total
                for k2 in counts:
                        if k1 == k2: continue
                        p2 = float( counts[k2] ) / total
                        imp += p1 * p2
        return imp


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
def classify(observation, tree):
        if tree.results != None:
                return tree.results
        else:
                v = observation[ tree.col ]
                if v == None:
                        #this means there is data missing
                        tr = mdclassify( observation, tree.tb )
                        fr = mdclassify( observation, tree.fb )

                        tcount = sum( tr.values() )
                        fcount = sum( fr.values() )

                        tw = float(tcount)/( tcount + fcount )
                        fw = float(fcount)/( tcount + fcount )

                        result = {}
                        for k,v in tr.items(): result[k] = v * tw
                        for k,v in fr.items(): results[k] = v * fw
                        return result
                else:
                        if isinstance(v, int) or isinstance(v, float):
                                if v >= tree.value: branch = tree.tb
                                else: branch = tree.fb
                        else:
                                if v==tree.value: branch = tree.tb
                                else: branch = tree.fb
                        return classify( observation, branch )

### Step 6 pruning the tree
def prune( tree, mingain ):
        #if branches are not leaves, then prune
        if tree.tb.results == None:
                prune( tree.tb, mingain )
        if tree.fb.results == None:
                prune( tree.fb, mingain )

        #if both subbranches are leaves, chech if they should merge
        if tree.tb.results != None and tree.fb.results != None:
                tb,fb = [],[]
                for v,c in tree.tb.results.items():
                        tb += [[v]]*c
                for v,c in tree.fb.results.items():
                        fb += [[v]]*c

                #test for reduction in entropy
                delta = entropy( tb + fb ) - (entropy(tb) + entropy(fb)/2)

                if delta < mingain:
                        #merge branches
                        tree.tb, tree.fb = None,None
                        tree.results = uniquecounts( tb + fb )

## Step 7 convert result to label
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

#####################################################
## KNN
def getdistances(data, query):
        #List to put the distances into.
        distancelist=[]

        continuous_dist = 0
        categorical_dist = 0

        # Iterate over every item in the dataset
        for i in range(len(data)):
                # Numerical Distance between between continuous data.
                continuous_dist = continuous_dist + math.sqrt((data[i][1]  - query[1])**2)
                continuous_dist = continuous_dist + math.sqrt((data[i][3]  - query[3])**2)
                continuous_dist = continuous_dist + math.sqrt((data[i][5]  - query[5])**2)
                continuous_dist = continuous_dist + math.sqrt((data[i][11] - query[11])**2)
                continuous_dist = continuous_dist + math.sqrt((data[i][12] - query[12])**2)
                continuous_dist = continuous_dist + math.sqrt((data[i][13] - query[13])**2)

                # Categorical distance between data and query
                if not(data[i][2] == query[2]):
                        categorical_dist += 1
                if not(data[i][4] == query[4]):
                        categorical_dist += 1
                if not(data[i][6] == query[6]):
                        categorical_dist += 1
                if not(data[i][7] == query[7]):
                        categorical_dist += 1
                if not(data[i][8] == query[8]):
                        categorical_dist += 1
                if not(data[i][9] == query[9]):
                        categorical_dist += 1
                if not(data[i][10] == query[10]):
                        categorical_dist += 1
                if not(data[i][14] == query[14]):
                        categorical_dist += 1

                # Add the distance and the index to the distance list
                distancelist.append(([data[i][0],continuous_dist + categorical_dist,data[i][15]]))
                
        #Sort Distance List
        distancelist.sort()
        return distancelist

def knnestimate(data,query,k=10):

        #Get the sorted distances between the query and each record in the data set
        dlist = getdistances(data,query)
        over50k_count = 0
        under50k_count = 0

        for i in range(k):
                if dlist[i][2] == ' <=50K':
                        under50k_count = under50k_count + 1
                else:
                        over50k_count  = over50k_count + 1

        
        # Category with highest number of records                
        if under50k_count > over50k_count:
                return ' <=50K'
        else:
                return ' >50K'

## KNN End
#####################################################

#####################################################
###                 Main Function                 ###
#####################################################
def main():

        #change this to switch between modes
        testMode = True

        #Enable/Disable Decision Tree or KNN prediction model 
        DecisionTree= True
        KNN =False

        #Testing Mode enabled
        if( testMode ):
                #Decision Tree Enabled
                if(DecisionTree):

                        #Step 1 Get data
                        trainingData = parseFile('./data/trainingset.txt')
                        trainingSet = splitListIntoEqualSize( trainingData, 500 )

                        # Step 2 train the model
                        tree = buildTree( trainingSet[0] )

                        # Step 3 prune the model
                        prune( tree, 0.1 )

                        #cross validation
                        for i in range( 1, len(trainingSet) ):

                                predictions = []
                                correct = 0

                                for j in range( len(trainingSet[i]) ):
                                        o = classify( trainingSet[i][j], tree )
                                        p = convertDecTreeOutput2Label(o)
                                        predictions.append(p)

                                for h in range( len(predictions) ):
                                        if predictions[h] == trainingSet[i][h][-1]:
                                                correct += 1
                                print("Percentage correct: " + str("%.4f" % round(float(correct)/len(predictions),4)))
                #KNN Enabled
                if(KNN):
                        #Step 1 Get data
                        testSet = parseFile('./data/queries.txt')
                        trainingData = parseFile('./data/trainingset.txt')

                        correctPrediction =[]

                        #Step 2 Retrieve Correct Prediction
                        for i in range(len(trainingData)):
                                result = trainingData[i][15]
                                correctPrediction.append(result)

                        correct = 0
                        incorrect = 0

                        results=[]
                        #Step 3 Make Prediction
                        for i in range(len(testSet)):
                                estimate = knnestimate(trainingData,testSet[i])

                                # Step 4 Compare Prediction to Correct Answer
                                if estimate == correctPrediction[i]:
                                        correct += 1
                                else:
                                        incorrect += 1
                        #Step 5 Calculate Accuracy of KNN Model
                        print(str("Percentage correct: " + "%.4f" % round(float(correct)/(correct+incorrect),4)))
                        

                       
        # Testing Mode Disabled
        # Generate Solution
        else:

                print("working...")

                #Step 1 Get data
                queryData = parseFile('./data/queries.txt')
                trainingData = parseFile('./data/trainingset.txt')

                predictions = []

                # Decision Tree Enabled
                if(DecisionTree):
                        
                        trainingSet = splitListIntoEqualSize( trainingData, 1000 )

                        # Step 2 train the model
                        tree = buildTree( trainingSet[0] )

                        # Step 3 prune the model
                        prune( tree, 0.1 )

                      

                        for i in range( len(queryData) ):
                                o = classify( queryData[i], tree )
                                p = convertDecTreeOutput2Label( o )
                                predictions.append( p )
                # KNN Enabled                
                if(KNN):
                        for i in range(len(queryData)):
                                estimate = knnestimate(trainingData,queryData[i])
                                predictions.append(estimate)

                # Generate Solutions File
                resultsFile = open('./solutions/C10319445+C10736831.txt','w+')

                for i in range( len(predictions) ):

                        resultsFile.write( queryData[i][0].strip() )
                        resultsFile.write( "," )
                        resultsFile.write( predictions[i].strip() )
                        resultsFile.write( "\n" )

                resultsFile.close()
                print("done...")

main()
