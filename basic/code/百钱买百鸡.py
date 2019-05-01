'''
我国古代数学家张丘建在《算经》一书中提出的数学问题：鸡翁一值钱五，鸡母一值钱三，
鸡雏三值钱一。百钱买百鸡，如果要求鸡翁、鸡母、鸡雏都不为零，问鸡翁、鸡母、鸡雏各几何？‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‮‬‭‬‫‬‪‬‪‬‪‬‪‬‪‬‮‬‪‬‪‬‪‬‪‬‪‬‪‬‪‬‮‬‫‬‮‬
'''
rooster = 5
hen = 3
chick = 1 / 3
lst = []
for roo in range(1,20):
    for he in range(1,33):
        for chi in range(3,300):
            if int(roo*rooster+hen*he+chi*chick) == 100 and chi % 3 == 0 and roo + he + chi == 100:
                lst.append((roo,he,chi))
for x in lst:
    print("{} {} {}".format(x[0],x[1],x[2]))