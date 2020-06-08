
# fullname = "{in1} {in2}".format(in1 =fname, in2=lname)
# # print(fullname)
# # name = " ".join(fullname)
# # name = (str)fullname
# # fname.append(lname
# # print(fname+lname)
# def listtostring(s):
#      stgr=""
#      return(stgr.join(s))
# # print(listtostring(fname))
#
# fullname=fs+" "+ls

fname = ['prajwal']
lname =['mani']
def listtostring(s):
    stgr = ""
    return (stgr.join(s))
fullname = listtostring(fname)+" "+listtostring(lname)
print(fullname)
