test = [1,2,3,4,5,6,-1,-1]
for i in range(len(test)):
    if test[i] == -1:
        test[i] = 0 


print(test)