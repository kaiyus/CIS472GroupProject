from __future__ import division
import sys
import re
import math

def read_data(filename):
	f = open(filename, 'r')
	p = re.compile(',')
	data = []
	company = []
	barname = []
	ref = []
	reviewdata = []
	cocoa =[]
	location = []
	rating = []
	beantype = []
	broad = []
	print (f.readline().strip())
	print (f.readline().strip())
	print (f.readline().strip())
	print (f.readline().strip())
	print (f.readline().strip())
	print (f.readline().strip())
	print (f.readline().strip())
	print (f.readline().strip())
	namehash = {}
	for l in f:
		datalist = p.split(l.strip())
		broad.append(datalist.pop())
		company.append(datalist.pop(0))
		r = 0;
		for i in range(0,len(datalist)):
			if datalist[i].isdigit():
				r = i
				break;
		ref.append(datalist.pop(r))
		reviewdata.append(datalist.pop(r))
		cocoa.append(datalist.pop(r))
		location.append(datalist.pop(r))
		rating.append(datalist.pop(r))
		bl = len(datalist) - r
		be = []
		for i in range(0,bl):
			be.append(datalist.pop())
		beantype.append(be)
		barname.append(datalist)

	companyvar = makevarnames1(company)
	companyresult = turndataintobinary1(company, companyvar)
	broadvar = makevarnames1(broad)
	broadresult = turndataintobinary1(broad, broadvar)
	locationvar = makevarnames1(location)
	locationresult = turndataintobinary1(location, locationvar)
	beantypevar = makevarnames2(beantype)
	beantyperesult = turndataintobinary2(beantype, beantypevar)
	barnamevar = makevarnames2(barname)
	barnameresult = turndataintobinary2(barname, barnamevar)

	result = []
	for i in range(len(company)):
		xlist = companyresult[i] 
		xlist += barnameresult[i] 
		xlist += [ref[i]] 
		xlist += [reviewdata[i]] 
		xlist += [cocoa[i]] 
		xlist += locationresult[i] 
		xlist += beantyperesult[i] 
		xlist += broadresult[i]
		result.append((xlist,rating[i]))
	return result


def turndataintobinary1(data, varname):
	l = len(varname)
	result = []
	for i in range(0,len(data)):
		thisdata = ([0] * l)
		thisdata[varname.index(data[i])] = 1
		result.append(thisdata)
	return (result)

def turndataintobinary2(data, varname):
	l = len(varname)
	result = []
	for i in range(0,len(data)):
		thisdata = ([0] * l)
		for j in range(0,len(data[i])):
			thisdata[varname.index(data[i][j])] = 1
		result.append(thisdata)
	return (result)

def makevarnames1(data):
	var = []
	for i in range(0,len(data)):
		if data[i] not in var:
			var.append(data[i])
	return var

def makevarnames2(data):
	var = []
	for i in range(0,len(data)):
		for j in range(0,len(data[i])):
			if data[i][j] not in var:
				var.append(data[i][j])
	return var


def main(argv):
	data = read_data(argv[0])
	print (data)
                    
if __name__ == "__main__":
	main(sys.argv[1:])