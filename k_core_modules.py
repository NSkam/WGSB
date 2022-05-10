import math
from math import log

import numpy
numpy.set_printoptions(threshold=numpy.inf)

import sys

import nltk
from nltk.corpus import stopwords
import collections
import matplotlib.pyplot as plt
import networkx as nx

import string
translator = str.maketrans('', '', string.punctuation)

from operator import itemgetter


from networkx import core_number, k_core
import csv
#buggy na xrisimopoii8ei i proepeksergasia pou exei dimiourgi8ei san ksexwristo .py
postinglist = []
docinfo = []
docs_without_main_core =[]
def preproccess(file):
    with open(file, 'r') as fd:
        text = fd.read().split()
    fd.close()
    with open(file, 'w'):
        pass
    with open(file, 'a') as fd:
        fd.write('')
        for term in text:
            term = term.translate(translator)
            term = term.upper()
            if len(term) != 1:
                fd.write("%s \n" % term)
    fd.close()
    return 1

#Split File in smaller files according to window size.
#If window size is equal to zero the function calculates
#the window by taking into account the total length of the
#file. (minimum window = 5)
def splitFileConstantWindow(file,window,per_window):
    	
    #Open the file and split it into words
    inputFile = open(file, 'r').read().split()
    num_of_words = len(inputFile)
    outputFile =[]
	
	#If window is equal to zero get window according to length or if percentage window flag is true
    if window == 0:
        #print(per_window)
        window = int(num_of_words * per_window) + 1
        #print("Window Size: ", window)
        if window <5:
           window =5
	
    #Join words according to window
    for i in range(0,num_of_words,window):
        outputFile.append(' '.join(inputFile[i:i+window]))
    
    #print(outputData)
    return outputFile

def createInvertedIndexFromFile(file, postingl):

    with open(file, 'r') as fd:
    # list containing every word in text document
        text = fd.read().split()
        uninque_terms = []
        termFreq = []
        for term in text:
            if term not in uninque_terms:
                uninque_terms.append(term)
                termFreq.append(text.count(term))
            if term not in postingl:
                postingl.append(term)
                postingl.append([file, text.count(term)])
            else:
                existingtermindex = postingl.index(term)
                if file not in postingl[existingtermindex + 1]:
                    postingl[existingtermindex + 1].extend([file, text.count(term)])
    # print(len(uninque_terms))
    # print(termFreq)
    return (uninque_terms, termFreq, postingl, len(text))

    ###############################lemmas################################
    # Weight_of_edge(i,j) = No.occurencies_of_i * No.occurencies_of_j   #
    #####################################################################
    
	#First we split the file into an array, in which each element represents a part of the file
	#with length equal to window_size. We then take each part and create the adjacency matrix.
	#Each element a(i,j), where i,j are two different terms of the file, is equal to the number
	#of times the term i and the term j appear in the same window.
def CreateAdjMatrixFromInvIndexWithWindow(terms, file, window_size,per_window,dot_split):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    #print(terms)
    adj_matrix = numpy.zeros(shape=(len(terms),len(terms)))
    tfi=0
    if dot_split:
        inputFile = open(file, 'r').read()
        split_file = nltk.tokenize.sent_tokenize(inputFile, language='english')
    else:
        split_file = splitFileConstantWindow(file, window_size,per_window)
    for subfile in split_file:
        window_terms = subfile.split()
        for term in window_terms:
            #print("\n")
            #print(term)
            row_index = terms.index(term)
            #print("TERM:",row_index)
            for x in range(0,len(window_terms)):
                col_index = terms.index(window_terms[x])
                #print("Y TERM:",col_index)
                if col_index == row_index:
                    tfi+=1
                else:
                    adj_matrix[row_index][col_index]+=1
            adj_matrix[row_index][row_index]+=tfi*(tfi+1)/2
            tfi=0
    
    #pen_adj_mat = applyStopwordPenalty(adj_matrix, terms)
    #print(adj_matrix)
    #fullsize = rows.size + row.size + col.size + adj_matrix.size
    #print(fullsize / 1024 / 1024)
    return (adj_matrix)
    
def applyStopwordPenalty(adj_matrix, term_list):

    stopword = ['A', 'ABOUT', 'ABOVE', 'ACROSS', 'ACTUALLY', 'ADD', 'ADDED', 'AFTER', 'AGAIN', 'AGAINST', 'AGO', 'ALL', 'ALMOST', 'ALONG', 'ALREADY', 'ALSO', 'ALTHOUGH', 'ALWAYS', 'AM', 'AMONG', 'AN', 'AND', 'ANOTHER', 'ANY', 'ANYONE', 'ARE', 'AROUND', 'AS', 'ASKED', 'AT', 'B', 'BACK', 'BAD', 'BE', 'BECAME', 'BECAUSE', 'BECOME', 'BEEN', 'BEFORE', 'BEGAN', 'BEHIND', 'BEING', 'BEST', 'BETTER', 'BETWEEN', 'BIG', 'BIGGEST', 'BOTH', 'BROUGHT', 'BUT', 'BY', 'C', 'CALLED', 'CAME', 'CAN', 'CANNOT', 'CENT', 'COME', 'COMPLETE', 'CONTINUED', 'COULD', 'D', 'DAY', 'DECIDED', 'DECLARED', 'DESPITE', 'DID', 'DO', 'DOES', 'DOWN', 'DURING', 'E', 'EACH', 'EARLY', 'EIGHT', 'ENOUGH', 'ENTIRE', 'EP', 'ETC', 'EVEN', 'EVER', 'EVERY', 'EVERYTHING', 'F', 'FACE', 'FACED', 'FACT', 'FAILED', 'FAR', 'FELL', 'FEW', 'FINALLY', 'FIND', 'FIRST', 'FIVE', 'FOR', 'FOUND', 'FOUR', 'FROM', 'G', 'GAVE', 'GET', 'GIVE', 'GIVEN', 'GO', 'GOING', 'GOOD', 'GOT', 'H', 'HAD', 'HAS', 'HAVE', 'HAVING', 'HE', 'HELD', 'HER', 'HERE', 'HIM', 'HIMSELF', 'HIS', 'HOUR', 'HOURS', 'HOW', 'HOWEVER', 'I', 'IDEA', 'IF', 'IN', 'INCLUDING', 'INSTEAD', 'INTO', 'IS', 'IT', 'ITS', 'ITSELF', 'J', 'K', 'KEEP', 'KNOW', 'KNOWN', 'KNOWS', 'L', 'LACK', 'LAST', 'LATER', 'LEAST', 'LED', 'LESS', 'LET', 'LIKE', 'LITTLE', 'LONG', 'LONGER', 'LOOK', 'LOT', 'M', 'MADE', 'MAKE', 'MAKING', 'MAN', 'MANY', 'MATTER', 'MAY', 'ME', 'MEANS', 'MEN', 'MIGHT', 'MILES', 'MILLION', 'MOMENT', 'MONTH', 'MONTHS', 'MORE', 'MORNING', 'MOST', 'MUCH', 'MUST', 'MY', 'N', 'NAMED', 'NEAR', 'NEARLY', 'NECESSARY', 'NEED', 'NEEDED', 'NEEDS', 'NEVER', 'NIGHT', 'NO', 'NOR', 'NOT', 'NOTE', 'NOTHING', 'NOW', 'O', 'OF', 'OFF', 'OFTEN', 'ON', 'ONCE', 'ONE', 'ONLY', 'OR', 'OTHER', 'OTHERS', 'OUR', 'OUT', 'OUTSIDE', 'OVER', 'OWN', 'P', 'PAGE', 'PART', 'PAST', 'PER', 'PERHAPS', 'PLACE', 'POINT', 'PROVED', 'PUT', 'Q', 'QM', 'QUESTION', 'R', 'REALLY', 'RECENT', 'RECENTLY', 'REPORTED', 'ROUND', 'S', 'SAID', 'SAME', 'SAY', 'SAYS', 'SEC', 'SECOND', 'SECTION', 'SEE', 'SEEMED', 'SEEMS', 'SENSE', 'SET', 'SETS', 'SEVEN', 'SHE', 'SHORT', 'SHOULD', 'SHOWED', 'SINCE', 'SINGLE', 'SIX', 'SMALL', 'SO', 'SOME', 'SOON', 'START', 'STARTED', 'STILL', 'SUCH', 'T', 'TAKE', 'TAKEN', 'TAKES', 'TEN', 'TEXT', 'THAN', 'THAT', 'THE', 'THEIR', 'THEM', 'THEMSELVES', 'THEN', 'THERE', 'THESE', 'THEY', 'THING', 'THINGS', 'THIRD', 'THIS', 'THOSE', 'THOUGH', 'THOUGHT', 'THOUSANDS', 'THREE', 'THROUGH', 'THUS', 'TIME', 'TINY', 'TO', 'TODAY', 'TOGETHER', 'TOLD', 'TOO', 'TOOK', 'TOWARD', 'TWO', 'U', 'UNDER', 'UNTIL', 'UP', 'UPON', 'US', 'USE', 'USED', 'V', 'VERY', 'W', 'WARNING', 'WAS', 'WAY', 'WE', 'WEEK', 'WEEKS', 'WELL', 'WENT', 'WERE', 'WHAT', 'WHEN', 'WHERE', 'WHETHER', 'WHICH', 'WHILE', 'WHO', 'WHOM', 'WHOSE', 'WHY', 'WILL', 'WITH', 'WITHOUT', 'WORD', 'WORDS', 'WOULD', 'X', 'Y', 'YEAR', 'YEARS', 'YET', 'YOU', 'YOUR', 'Z']
    stopword_list = [x.upper() for x in stopword]
    penalized_adj_matrix = numpy.zeros(shape=(len(term_list),len(term_list))) 
     
    for term in term_list:
        #print(term)
        row_index = term_list.index(term)
        if term in stopword_list:
            for y in range(0,len(term_list)):
                penalized_adj_matrix[row_index][y] = adj_matrix[row_index][y]*1
            #print("Term " + str(term) + ": " + str(penalized_adj_matrix[row_index][:]))
            #print("Term " + str(term) + ": " + str(adj_matrix[row_index][:]))
        else:   
            for y in range(0,len(term_list)):
                penalized_adj_matrix[row_index][y] = adj_matrix[row_index][y]      
            continue
           
    #print(adj_matrix)
    #array_equality = numpy.array_equal(adj_matrix, penalized_adj_matrix)
    #print(array_equality)    
     
    return penalized_adj_matrix

	#The idea behind this function is to get two adjacency matrices(one for sentences and one for paragraphs)
	#using CreateAdjMatrixFromInvIndexWithWindow and then combine them into a single matrix using two weight 
	#coefficients a and b, which will determine the importance of each matrix.
def CreateAdjMatrixFromInvIndexWithSenParWindow(terms,file,sen_window_size, par_window_size,dot_split):

	matrix_size = len(terms)
	
	#Create the matrices
	sen_adj_matrix = numpy.zeros(shape=(matrix_size,matrix_size,))
	par_adj_matrix = numpy.zeros(shape=(matrix_size,matrix_size))
	
	#Get the adjacency matrix for each window
	sen_adj_matrix = CreateAdjMatrixFromInvIndexWithWindow(terms,file,sen_window_size,0,dot_split)
	par_adj_matrix = CreateAdjMatrixFromInvIndexWithWindow(terms,file,par_window_size,0,dot_split)
	
	#Create the final Matrix
	final_adj_matrix = numpy.zeros(shape=(matrix_size,matrix_size))
	
	#Create coefficients a and b
	a = 1.0
	b = 0.05
	
	#Add the two matrices
	final_adj_matrix = [[a*sen_adj_matrix[r][c] + b*par_adj_matrix[r][c]  for c in range(len(sen_adj_matrix[0]))] for r in range(matrix_size)]
	#print(final_adj_matrix)
	
	return final_adj_matrix

# using as an input the terms and the term frequency it creates the adjacency matrix of the graph
# in the main diagon we have the Win of each node of the graph and by the sum of each colume
# except the element of the diagon  is the  Wout of each node
# For more info see LEMMA 1 and LEMMA 2 of P: A graph based extension for the Set-Based Model, A: Doukas-Makris
def CreateAdjMatrixFromInvIndex(terms, tf):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    rows = numpy.array(tf)
    row = numpy.transpose(rows.reshape(1, len(rows)))
    col = numpy.transpose(rows.reshape(len(rows), 1))
    adj_matrix = numpy.array(numpy.dot(row, col))
    #fullsize = rows.size + row.size + col.size + adj_matrix.size
    #print(fullsize / 1024 / 1024)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if i == j:
                adj_matrix[i][j] = rows[i] * (rows[i] + 1) * 0.5
    #print(adj_matrix)
    del row, rows, col
    return (adj_matrix)

    ################################################################################################
    # For each node we calculate the sum of the elements of the respective row or colum of its index#
    # as its degree                                                                                 #
    ################################################################################################

def BucketHash(file_list):

    bucket_0 = []
    bucket_1 = []
    bucket_2 = []
    bucket_3 = []
    bucket_4 = []
    bucket_5 = [] 
    bucket_6 = []
    bucket_7 = []      

    for name in file_list:
        name = name[0]
        name = int(name[9:])
        hashed_name = name %6
        if hashed_name == 0:
            bucket_0.append(name)
        elif hashed_name == 1:
            bucket_1.append(name)
        elif hashed_name == 2:
            bucket_2.append(name)
        elif hashed_name == 3:
            bucket_3.append(name)
        elif hashed_name == 4:
            bucket_4.append(name)
        elif hashed_name == 5:
            bucket_5.append(name)
        elif hashed_name == 6:
            bucket_6.append(name)
        elif hashed_name == 7:
            bucket_7.append(name)    
        else:
            print("Something went wrong\n")
    return bucket_0, bucket_1, bucket_2, bucket_3, bucket_4, bucket_5, bucket_6, bucket_7

def calculateSummationMatrix(adj_matrix, file,file_sum_mat,terms, window):
    
    adj_mat_sum = adj_matrix.sum(axis = 1, dtype = 'float')
    for i in range(0,len(adj_mat_sum)):
        #print(adj_mat_sum[i])
        adj_mat_sum[i] = float(adj_mat_sum[i]/(window-1))
        #print(adj_mat_sum[i])
        if isinstance( adj_mat_sum[i], float):
            adj_mat_sum[i] = math.ceil(adj_mat_sum[i])
    #print(adj_mat_sum)
    sum_mat = list(zip(terms,adj_mat_sum))
    #print(sum_mat)
    sorted_sum_mat = sorted(sum_mat, key = lambda x: x[1],reverse=True)
    #print(sorted_sum_mat)
    file_sum_mat.append([file,sorted_sum_mat])
    #print(file_sum_mat)
    return file_sum_mat
    
def calculateStopwordWeight(file_sum_mat, collection_terms):

    a = 5
    
    collection_sum_mat = numpy.zeros(shape=(1,len(collection_terms)))
    collection_vote_mat = numpy.zeros(shape=(1,len(collection_terms)))
    weight_matrix = numpy.zeros(shape=(1,len(collection_terms)))
    #print(file_mat)

    for i in range(1,len(file_sum_mat)):
        file_terms_mat = file_sum_mat[i][1]
        #print(file_sum_mat[i][1])
        collection_sum_mat, collection_vote_mat = calculateWeight(collection_sum_mat, collection_vote_mat, file_terms_mat,collection_terms) #collection_matrices[0] = collection_sum_mat | collection_matrices[1] = collection_vote_mat
    #print(collection_sum_mat)
    #print( collection_vote_mat)
    for x in range(0,len(collection_sum_mat[0][:])):
        weight_matrix[0][x] = collection_sum_mat[0][x] + collection_vote_mat[0][x]*a
        
    indices = [x for x in range(len(weight_matrix[0][:]))]
    #print(weight_matrix)
    zipped_weight_matrix = list(zip(indices,*weight_matrix))
    sorted_weight_matrix = sorted(zipped_weight_matrix, key = lambda x: x[1],reverse=True)
    #print(sorted_weight_matrix)
    
    return sorted_weight_matrix

def calculateWeight(collection_sum_mat, collection_vote_mat, file_terms_mat, collection_terms):
    

    for x in range(0,len(file_terms_mat)):
        index = collection_terms[file_terms_mat[x][0]]
        #print(file_terms_mat[x][0])
        #print(index)
        collection_sum_mat[0][index] += file_terms_mat[x][1]
        #print(collection_sum_mat[0][index])
        collection_vote_mat[0][index] +=1
        #print(collection_vote_mat[index])
    
    #print(collection_sum_mat)
    
    return collection_sum_mat, collection_vote_mat

def stopwordsStats(stopword_matrix, collection_terms):

    stopwords = ['A', 'ABOUT', 'ABOVE', 'ACROSS', 'ACTUALLY', 'ADD', 'ADDED', 'AFTER', 'AGAIN', 'AGAINST', 'AGO', 'ALL', 'ALMOST', 'ALONG', 'ALREADY', 'ALSO', 'ALTHOUGH', 'ALWAYS', 'AM', 'AMONG', 'AN', 'AND', 'ANOTHER', 'ANY', 'ANYONE', 'ARE', 'AROUND', 'AS', 'ASKED', 'AT', 'B', 'BACK', 'BAD', 'BE', 'BECAME', 'BECAUSE', 'BECOME', 'BEEN', 'BEFORE', 'BEGAN', 'BEHIND', 'BEING', 'BEST', 'BETTER', 'BETWEEN', 'BIG', 'BIGGEST', 'BOTH', 'BROUGHT', 'BUT', 'BY', 'C', 'CALLED', 'CAME', 'CAN', 'CANNOT', 'CENT', 'COME', 'COMPLETE', 'CONTINUED', 'COULD', 'D', 'DAY', 'DECIDED', 'DECLARED', 'DESPITE', 'DID', 'DO', 'DOES', 'DOWN', 'DURING', 'E', 'EACH', 'EARLY', 'EIGHT', 'ENOUGH', 'ENTIRE', 'EP', 'ETC', 'EVEN', 'EVER', 'EVERY', 'EVERYTHING', 'F', 'FACE', 'FACED', 'FACT', 'FAILED', 'FAR', 'FELL', 'FEW', 'FINALLY', 'FIND', 'FIRST', 'FIVE', 'FOR', 'FOUND', 'FOUR', 'FROM', 'G', 'GAVE', 'GET', 'GIVE', 'GIVEN', 'GO', 'GOING', 'GOOD', 'GOT', 'H', 'HAD', 'HAS', 'HAVE', 'HAVING', 'HE', 'HELD', 'HER', 'HERE', 'HIM', 'HIMSELF', 'HIS', 'HOUR', 'HOURS', 'HOW', 'HOWEVER', 'I', 'IDEA', 'IF', 'IN', 'INCLUDING', 'INSTEAD', 'INTO', 'IS', 'IT', 'ITS', 'ITSELF', 'J', 'K', 'KEEP', 'KNOW', 'KNOWN', 'KNOWS', 'L', 'LACK', 'LAST', 'LATER', 'LEAST', 'LED', 'LESS', 'LET', 'LIKE', 'LITTLE', 'LONG', 'LONGER', 'LOOK', 'LOT', 'M', 'MADE', 'MAKE', 'MAKING', 'MAN', 'MANY', 'MATTER', 'MAY', 'ME', 'MEANS', 'MEN', 'MIGHT', 'MILES', 'MILLION', 'MOMENT', 'MONTH', 'MONTHS', 'MORE', 'MORNING', 'MOST', 'MUCH', 'MUST', 'MY', 'N', 'NAMED', 'NEAR', 'NEARLY', 'NECESSARY', 'NEED', 'NEEDED', 'NEEDS', 'NEVER', 'NIGHT', 'NO', 'NOR', 'NOT', 'NOTE', 'NOTHING', 'NOW', 'O', 'OF', 'OFF', 'OFTEN', 'ON', 'ONCE', 'ONE', 'ONLY', 'OR', 'OTHER', 'OTHERS', 'OUR', 'OUT', 'OUTSIDE', 'OVER', 'OWN', 'P', 'PAGE', 'PART', 'PAST', 'PER', 'PERHAPS', 'PLACE', 'POINT', 'PROVED', 'PUT', 'Q', 'QM', 'QUESTION', 'R', 'REALLY', 'RECENT', 'RECENTLY', 'REPORTED', 'ROUND', 'S', 'SAID', 'SAME', 'SAY', 'SAYS', 'SEC', 'SECOND', 'SECTION', 'SEE', 'SEEMED', 'SEEMS', 'SENSE', 'SET', 'SETS', 'SEVEN', 'SHE', 'SHORT', 'SHOULD', 'SHOWED', 'SINCE', 'SINGLE', 'SIX', 'SMALL', 'SO', 'SOME', 'SOON', 'START', 'STARTED', 'STILL', 'SUCH', 'T', 'TAKE', 'TAKEN', 'TAKES', 'TEN', 'TEXT', 'THAN', 'THAT', 'THE', 'THEIR', 'THEM', 'THEMSELVES', 'THEN', 'THERE', 'THESE', 'THEY', 'THING', 'THINGS', 'THIRD', 'THIS', 'THOSE', 'THOUGH', 'THOUGHT', 'THOUSANDS', 'THREE', 'THROUGH', 'THUS', 'TIME', 'TINY', 'TO', 'TODAY', 'TOGETHER', 'TOLD', 'TOO', 'TOOK', 'TOWARD', 'TWO', 'U', 'UNDER', 'UNTIL', 'UP', 'UPON', 'US', 'USE', 'USED', 'V', 'VERY', 'W', 'WARNING', 'WAS', 'WAY', 'WE', 'WEEK', 'WEEKS', 'WELL', 'WENT', 'WERE', 'WHAT', 'WHEN', 'WHERE', 'WHETHER', 'WHICH', 'WHILE', 'WHO', 'WHOM', 'WHOSE', 'WHY', 'WILL', 'WITH', 'WITHOUT', 'WORD', 'WORDS', 'WOULD', 'X', 'Y', 'YEAR', 'YEARS', 'YET', 'YOU', 'YOUR', 'Z']   
    stopword_count = 0
    #print(len(stopword_matrix))
    #print(len(collection_terms))
    val_list = list(collection_terms.values())
    key_list = list(collection_terms.keys())
    new_matrix = stopword_matrix[0:1000][0:1000]
    for i in range(0,1000):
        #print(i)
        position = val_list.index(new_matrix[i][0])
        if key_list[position] in stopwords:
            stopword_count += 1
            #print(stopword_count)
    stopwords_in_collection = float(stopword_count/340)
    stopwords_per = float(stopword_count/len(new_matrix))
    #print(stopwords_per)
    string_to_write = " stopwords in matrix percentage : " + str(stopwords_per) + " and stopwords percentage in collection: "+ str(stopwords_in_collection) + " sta 1000\n"
    print(string_to_write)
    

# computes the degree of every node using adj matrix
def Woutdegree(mat):
    list_of_degrees = numpy.sum(mat, axis=0)
    list_of_degrees = numpy.asarray(list_of_degrees)
    id = []
    # print(list_of_degrees)
    # print(numpy.size(list_of_degrees))
    for k in range(numpy.size(list_of_degrees)):
        id.append(k)
        list_of_degrees[k] -= mat[k][k]
    list_of_degrees.tolist()
    return list_of_degrees, id


def sortByDegree(val):
    return val[0]


# todo: more efficient way to calculate max length of path it doesnt work on realistic scale (CANT BE DONE BECAUSE THE COMPLEXITY)
# finds the maximum distance which exists in the graph
def findMaxDistance(gr, adjmatrix):
    maxlist = []
    for adi in range(len(adjmatrix)):

        for adj in range(len(adjmatrix)):

            if adj < adi:
                # cut down the number of calculated paths path(i,j) == path(j,i) so we need only the upper
                # or lower  tri of adj_matrix
                path = list(nx.shortest_simple_paths(gr, adj, adi, weight='weight'))
                print('Longest Path for (%d,%d) is: ' % (adj, adi))
                print(path[-1])
                i = 0
                weightsum = 0
                for item in range(len(path[-1]) - 1):
                    indexI = path[-1][i]
                    indexIpp = path[-1][i + 1]
                    i += 1
                    weightsum += adjmatrix[indexI][indexIpp]
                maxlist.append(weightsum)
    print(len(maxlist))
    return max(maxlist)


# computes the cosine similarity of 2 vectors
def cos_sim(a, b):
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    if numpy.all(a == 0) or numpy.all(b == 0):
        ret = 0
    else:
        dot_p = numpy.dot(a, b)
        normA = numpy.linalg.norm(a)
        normB = numpy.linalg.norm(b)
        ret = dot_p / (normA * normB)
    return ret


# finds the maximum and the minimum similarity between the nodes of the graph
def node_simi(adjmatrix):
    max = 0
    min = 1
    for adi in range(len(adjmatrix)):
        for adj in range(len(adjmatrix)):
            if adj < adi:
                temp = cos_sim(adjmatrix[adi], adjmatrix[adj])
                if temp > max:
                    max = temp
                if temp < min:
                    min = temp
    # print(max, min)
    return max, min




# deletes by re drawing the graph edges of the graph given a minimum similarity
def pruneGraphbySimilarity(aMatrix, pers, minsim, termName):
    g = nx.Graph()
    for i in range(len(aMatrix)):
        for j in range(len(aMatrix)):
            if i > j:
                if cos_sim(aMatrix[i], aMatrix[j]) > minsim - ((minsim * pers)):
                    g.add_node(termName[i], id=i)
                    g.add_node(termName[j], id=j)
                    g.add_edge(i, j, weight=aMatrix[i][j])
    Matrix = nx.adjacency_matrix(g)
    # print(Matrix)
    return g, Matrix.todense()


def graphUsingAdjMatrix(adjmatrix, termlist, *args, **kwargs):
    gr = nx.Graph()
    filename = kwargs.get('filename', None)
    if not filename:
        filename = 'Name not found!' #used when i want to visualize graphs with name

    for i in range(0, len(adjmatrix)):
        gr.add_node(i, term=termlist[i])
        for j in range(len(adjmatrix)):
            if i > j:
                if adjmatrix[i][j]!=0:
                     gr.add_edge(i, j, weight=adjmatrix[i][j])
    # graphToPng(gr,filename = filename)
    return gr


# ------------------Graph visualization---------------
def getGraphStats(graph,filename,graphPng, degreePng):

	if nx.is_connected(graph):
		print("IT IS CONNECTED")
	name = filename[10:]
	if graphPng:
		graphToPng(graph = graph, filename = str(name))
	if degreePng:
		plot_degree_dist(graph = graph, filename = str(name))
	
	
def graphToPng(graph, *args, **kwargs):
    options = {
        'node_color': 'yellow',
        'node_size': 50,
        'linewidths': 0,
        'width': 0.1,
        'font_size': 8,
    }
    
    filename = kwargs.get('filename', None)
    if not filename:
        filename = 'Union graph'
    plt.figure(filename, figsize=(17, 8))
    plt.suptitle(filename)
    pos_nodes = nx.circular_layout(graph)
    nx.draw(graph, pos_nodes, with_labels=True, **options)
    pos_atrs = {}
    for node, coords in pos_nodes.items():
        pos_atrs[node] = (coords[0], coords[1] + 0.01)

    node_attrs = nx.get_node_attributes(graph, 'term')
    cus_node_att = {}
    for node, attr in node_attrs.items():
        cus_node_att[node] = attr
    nx.draw_networkx_labels(graph, pos_atrs, labels=cus_node_att, font_color='red', font_size=8)

    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos_nodes, edge_labels=labels)
    #plt.show()
    plt.savefig('figures/allq/'+str(filename)+'.png',format="PNG", dpi=600)
	
	
def plot_degree_dist(graph, *args, **kwargs):

    filename = kwargs.get('filename', None)
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=3)
    ax.set_xticklabels(deg)

    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    pos = nx.spring_layout(graph)
    plt.axis("off")
    nx.draw_networkx_nodes(graph, pos, node_size=20)
    nx.draw_networkx_edges(graph, pos, alpha=0.4)
    
    plt.savefig('figures/allq/'+str(filename)+'_degree.png',format="PNG", dpi=600)
	

# -----------Union Graph to inverted index-------------
def graphToIndex(id, terms, calc_term_w, plist, *args, **kwargs):
    filename = kwargs.get('filename', None)
    if not filename:
        filename = 'inverted index.dat'
    f = open(filename, "a+")
    data = ','.join(
        [str(i) for i in plist])  # join list to a string so we can write it in the inv index and load it with ease
    f.write('%d;%s;%f;%s;\n' % (id, terms, calc_term_w, data))
    f.close()
    return 1


# creating Rational path Graph - Union Graph


# calculating the weight and write the inverted index file using graphToIndex method
# NO USE
def w_and_write_to_file(listofdeg, Umatrix, collection_terms, union_graph_termlist_id, collection_term_freq):
    print('here')
    for i in range(len(listofdeg[0])):
        Wout = listofdeg[0][i]
        Win = collection_term_freq[i]
        nbrs = numpy.count_nonzero(Umatrix[i])
        VarA = 1
        VarB = 10
        Alog = 1 + VarA * ((Wout / (nbrs + 1)) / (Win + 1))
        Blog = 1 + VarB * (1 / (nbrs + 1))
        temp = log(Alog) * log(Blog)
        print(temp)

        indexofw = postinglist.index(collection_terms[i])  # maybe not the best way of implementing the
        graphToIndex(union_graph_termlist_id[i], collection_terms[i], temp, postinglist[indexofw + 1])
    return 1


def w_and_write_to_filev2(wout, collection_terms, union_graph_termlist_id, collection_term_freq, postinglist, file):
    # wout |[[term][wout][neibours]]\
    print('Calculating weights and create inveted index')
    print(file)
    for entry in collection_terms:
        term = entry
        id = collection_terms[entry]
        win = collection_term_freq[id]
        for sublist in wout:
            if term in sublist[0]:
                # print('here')
                Wo = sublist[1]
                nbrs = sublist[2]
                break
        VarA = 1
        VarB = 10
        Alog = 1 + VarA * ((Wo / (nbrs + 1)) / (win + 1))
        Blog = 1 + VarB * (1 / (nbrs + 1))
        try:
            temp = log(Alog) * log(Blog)
        except:
            print("Temp:",temp,"\nAlog:",Alog,"\nBlog:",Blog)
        indexofwordinlist = 2 * id + 1
        graphToIndex(id, term, temp, postinglist[indexofwordinlist], filename=file)
    return 1


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def union(lista, listb):
    c = []
    for i in lista + listb:
        if i not in c:
            c.append(i)
    return c


# generate new k+1 itemsets !!!reference to: Fast Algorithms for Mining Association Rules by Argawal
# a nice change would be the closed sets idea => less sets than frequent termsets
def apriori(l1, minfreq):
    final_list = []
    final_list.append(l1)
    k = 2
    l = l1
    print('=========Generating frequent sets ============')
    while (l != []):
        print("Starting Gen")
        c = apriori_gen(l)
        print("Ended Gen\n Starting Prune")
        l = apriori_prune(c, minfreq)
        # print('Frequent  %d-termset is: %s'%(k,l))
        # print(len(l))
        final_list.append(l)
        # print('====for k = %d the l list is' %k )
        print(k)
        k += 1
    return final_list


def apriori_gen(itemset):
    candidate = []
    ck = []
    texts = []
    length = len(itemset)
    for i in range(length):
        ele = itemset[i][0]
        for j in range(i + 1, length):
            ele1 = itemset[j][0]
            # print(ele, ele1)
            if ele[0:len(ele) - 1] == ele1[0:len(ele1) - 1]:  # and ele1 != ele:
                texts.append(intersection(itemset[i][1], itemset[j][1]))
                candidate.append([union(ele, ele1), intersection(itemset[i][1], itemset[j][1])])
    return candidate


def apriori_prune(termsets_list, min_support):
    prunedlist = []
    for j in termsets_list:
        if len(j[1]) > min_support:
            # print('-----------')
            # print(j[0],len(j[1]))
            # print('-----------')
            prunedlist.append([j[0], j[1]])

    return prunedlist


def printmenu():
    # menu implementation
    print("1.create index file seperate graphs and  union graph")
    print("2.load index file and then quering \n \n")

    #x = input('Insert option: ')
    hargs = int(sys.argv[1])
    print(hargs)
    S =float(sys.argv[2])
    print(S)
    x = int(sys.argv[3])
    print(x)

    return x, hargs, S


def doc_rep(doc_vec, idf_vec, *args, **kwargs):
    args = list(args)
    if not args:
        nw = []
        for i in range(len(idf_vec)):
            nw.append(1)
    else:
        nw = args[0]
    # print(nw)
    test = numpy.zeros((len(doc_vec), len(idf_vec)))
    for i in range(len(doc_vec)):
        # print(docs[i])
        for j in range(len(idf_vec)):
            # print(doc_vec[i][j])
            if doc_vec[i][j] > 0:
                test[i][j] = (1 + log(doc_vec[i][j])) * idf_vec[j] * float(nw[j])
            else:
                test[i][j] = 0
    # with open('debuglog.dat', 'a') as fd:
    #    fd.write('doc representa \n')
    #   for doci in test:
    #        fd.write('%s \n' %str(len(doci)))
    # fd.close()
    return test


def load_inv_index(*args):
    arg = list(args)
    if not arg:
        invindex = 'inverted index.dat'
    else:
        invindex = arg[0]
    ids = []
    trms = []
    W = []
    plist = []
    plist_expanded = []
    with open(invindex, 'r') as csvf:
        reader = csv.reader(csvf, delimiter=";")
        for row in reader:
            if row[0] not in ids:
                ids.append(row[0])
                trms.append(row[1])
                W.append(row[2])
                plist.append(row[3].split(','))
                plist_expanded.append(row[1])
                plist_expanded.append(row[3].split(','))
    csvf.close()
    # print(len(ids))
    return ids, trms, W, plist, plist_expanded


def load_doc_info(*args):
    args = list(args)
    if not args:
        docinfofile = "docinfo.dat"
    else:
        docinfofile = args[0]
    info = []
    with open(docinfofile, "r") as fh:
        lines = [line.split() for line in fh]
        for line in lines:
            if line not in info:
                info.append(line)
    return info


# input the Query Q as a list of words consisting the Query
def one_termsets(Q, trms, plist, minfreq):
    termsets = []
    One_termsets = []
    for word in Q:
        if word in trms:
            i = trms.index(word)
            doc = plist[(i)]
            doc = doc[::2]
            #print(doc)
            word = [''.join(word)]
            if len(doc) > minfreq:
                One_termsets.append([word, doc])
        else:
            print('word %s has not required support or it already exists:' % word)
    return One_termsets


def fij_calculation(file_list, final_list, plist, trms):
    
    docs =[]
    doc_list = []
    weight_doc_matrix = []
    doc_vec = []
    
    #print(file_list)
    #print(final_list)
    #print("\n\n\n\n")
    #print(plist)
    #sprint(trms)
    for itemsetList in final_list:
        #print("i ======= %s"%i)
        #print(itemsetList)
        for itemset in itemsetList:
            #print(itemset)
            #print(file[0])
            print(len(file_list))
            for file in file_list:
                #print(file[0])
                docs.append(file[0])
                
                #print(itemset[1])
                if file[0] in itemset[1]:
                    #print(plist[1])
                    itemsetTerms = itemset[0]
                    #print(itemsetTerms)
                    for itemset_term in itemsetTerms:
                        #print(itemset_term)
                        for j in range(0,len(plist),1):
                            #print(plist[j])
                            #print(itemset_term)
                            #print(plist[j-1])
                            if itemset_term in plist[j]:
                                if file[0] in plist[j+1]:
                                    file_index = plist[j+1].index(file[0])
                                    #print("File index" + str(file_index))
                                    weight =  plist[j+1][file_index+1]
                                    #print(file_index)
                                    weight_doc_matrix.append(int(weight))
                                    #print(file[0])
                                    #print(weight)
                    #print(weight_doc_matrix)
                    doc_vec.append(min(weight_doc_matrix))
                    weight_doc_matrix = []
                else:
                    doc_vec.append(0)
            doc_list.append(doc_vec)
            doc_vec = []
        #print(doc_list)
    doc_list = numpy.transpose(doc_list)
    #print(doc_list)
    doc_list = doc_list.tolist()
    #print(doc_list)
    return docs, doc_list


def calculate_idf(termsetsL, numofdocs):
    print('=====calculating idfs============')
    idf_vector = []
    for ts in termsetsL:  # iterate based on the number of terms in termset
        for item in ts:  # iterate all termsets with the same number of terms in set
            Nt = len(item[1])
            N = numofdocs
            if Nt != 0:
                idf = log(1 + (N / Nt))
                idf_vector.append(idf)
            else:
                idf_vector.append(0)
                print(item[1], len(item[1]))
    return idf_vector


# doukas weight
def calculate_termset_W(termsetsL, W, terms):
    print("=======Calculating W as stated in Doukas- Makris ======")
    termset_W_vector = []
    for ts in termsetsL:  # iterate based on the number of terms in termset
        for item in ts:  # iterate all termsets with the same number of terms in set
            product = 1
            for term in item[0]:
                if term in terms:
                    tindx = terms.index(term)
                    weight = W[tindx]
                    product *= float(weight)
            termset_W_vector.append(product)

    return termset_W_vector


def q_D_similarities(q, docmatrix, docs):
    ret_list = []
    cnt = 0
    # debug
    # with open('debuglog.dat', 'a') as fd:
    #    fd.write('doc matrix \n')
    #   for doci in docmatrix:
    #       fd.write('%s \n' %str(len(doci)))
    # fd.close()
    for doci in docmatrix:
        # the 0 array issue is fixed inside the cosine similarity function
        try:
            temp = cos_sim(q, doci)
            ret_list.append([docs[cnt], temp])
            cnt += 1
        except ValueError:
            print(doci)
            print(len(doci))
            print(cnt)
            exit(-1)
    return ret_list


def Woutusinggraph(inputgraph):
    nd = inputgraph.nodes()
    woutlist = []
    for n in nd:
        # print(n,inputgraph.degree(n,weight= 'weight'))
        woutlist.append([n, inputgraph.degree(n, weight='weight'), len(list(inputgraph.neighbors(n)))])

    print('success')
    return woutlist

def density(A_graph):
    graph_edges = A_graph.number_of_edges()
    #print(graph_edges)
    graph_nodes = len(list(A_graph.nodes))
    #print(graph_nodes)
    dens = graph_edges/(graph_nodes*(graph_nodes-1))
    return dens
#given points A and B it caluclates the distance of a point P from the line AB
def distance_to_line(starting_point,end_point,point):
    dist = -9999
    #spoint = (x1,y1)
    x1=starting_point[0]
    y1=starting_point[1]
    #end point = (x2,y2)
    x2=end_point[0]
    y2=end_point[1]
    #point = (x0,y0)
    print(point)
    x0=point[0]
    y0=point[1]
    dist = (abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y1*x1))/(math.sqrt(((y2-y1)**2)+((x2-x1)**2)))

    return dist

def elbow(listofpoints):
    #at first we need to create a line between first and last element
    if len(listofpoints)==1:
        bestindex = 0
    elif len(listofpoints)==2:
        if listofpoints[0]>listofpoints[1]:
            bestindex = 0
        else:
            bestindex = 1
    elif len(listofpoints)>2:
        #p1 the starting point of line and p2 the last point of line
        #using that we will calulate the distance of each point of our starting list
        #from the line using the known forumla
        p1 = numpy.array([listofpoints[0],0])
        p2 = numpy.array([listofpoints[-1],(len(listofpoints)-1)])
        distance = []
        #print(p1,p2)
        #print(listofpoints)
        pnt = []
        for point in listofpoints:
            pnt.append(point)
            pnt.append(listofpoints.index(point)+1)
            print(pnt)
            distance.append(distance_to_line(p1,p2,pnt))
            pnt =[]
        bestdistance = max(distance)
        bestindex = distance.index(bestdistance)
    return bestindex
# test method
