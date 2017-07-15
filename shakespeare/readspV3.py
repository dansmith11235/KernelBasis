#libraries used
import sys
import glob, os
import pandas as pd
import re
from itertools import chain
import string
#import pandas.rpy.common as com


#takes in the directory from user iput
#arg1 = sys.argv[1]

#sets the directory
#os.chdir(arg1)

# creates an empty dictionary to add play titles and array of words
dict = {}

# iterate through each file of directory
for file in glob.glob("*.txt"):
	
	#open play file
	play = open(file,'r')
	#read play lines into variable
	playlines =  play.readlines()
	#extract play name
	playname = re.sub("[^a-zA-Z]","",playlines[1].replace(" ",""))
	
	#create an empty array
	arr = []
	
	#place holder for play name to be used when creating dictionary
	tempname = 'na'
	#counter used to be able to tell when the play ends
	counter = 0
	#iterate through play text file and extract an array of words for each act
	for line in playlines:
		counter = counter + 1
		if (re.search('^ACT (I+|V+)+$', line) != None) or (len(playlines) == counter):
			dict[tempname] = arr
			arr = []
			if (re.search('^ACT (I+|V+)+$', line) != None):
				tempname = playname + re.search('^ACT (I+|V+)+$', line).group(0).replace(" ","")
			
				
		arr.append(line.split())
	
	#iterates through the play text file with no acts
	arr = []
	test = True
	for line in playlines: 
		test = test & (re.search('^ACT (I+|V+)+$', line) == None) 
		arr.append(line.split())
	if test == True:
		dict[playname] = arr
	
	
#remove "na" key from dictionary
dict.pop("na")

#create empty dictionary to record features 
dict2 = {}
#iterate through each item in the dictionary
for key in dict:

	
	
	#creates the counters that are used for the five ratios
	numDo = 0
	numDid = 0
	numNo = 0
	numNot = 0
	numThe = 0
	numTo = 0
	numUpon = 0
	numOn = 0
	numTten = 0
	
	for word in list(chain.from_iterable(dict[key])):
		
		w = word.lower()
		
		if w == "do":
			numDo = numDo + 1
		if w == "did":
			numDid = numDid + 1
		if w == "no":
			numNo = numNo + 1
		if w == "not":
			numNot = numNot + 1
		if w == "the":
			numThe = numThe + 1
		if w == "to":
			numTo = numTo + 1
		if w == "upon":
			numUpon = numUpon + 1
		if w == "on":
			numOn = numOn + 1
		if (
			w == "but" or w == "by" or w == "for" or w == "no" or w == "not" or w == "so"
			or w == "that" or w == "the" or w == "to" or w == "with"
			):
			numTten = numTten + 1
	arr = [(numDo/(numDid + numDo)),(numNo/(numTten)),(numNo/(numNo+numNot)),(numThe/(numTo + numThe)),(numUpon/(numOn+numUpon))]
	#rr = [numDo, numDid,numNo,numNot,numThe,numTten,numTo,numUpon,numOn]
	dict2[key] = arr

df = pd.DataFrame(dict2) #columns = ["did/do+did","no/T10","no/no+not","to/the+to","upon/on+upon"])
finaldf = df.T
finaldf.columns = ['DoByDidandDo','NobyTten','NobyNotando','ThebyToandThe','UponbyOnandUpon']


finaldf.to_csv("shakespeare.txt")
	

