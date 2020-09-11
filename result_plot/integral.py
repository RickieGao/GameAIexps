def mute(x, y, xlimit):
    coordinate = list(zip(x, y))
    coordinate = [(x, y) for (x, y) in coordinate if x <= xlimit]
    return coordinate


x1 = [1, 2, 3, 5, 6]
y1 = [10, 20, 30, 50, 70]
xlimit1 = 5
# coordinate1 = zip(x1, y1)
# list1 =list(coordinate1)
# for i in range(0, len(list1)):
#     print(list1[i])
# result = [(x, y) for (x, y) in list1 if x <= xlimit1]
result = mute(x1, y1, xlimit1)
print(result)
x2 = [x[0] for x in result]
y2 = [x[1] for x in result]
print(x2)
print(y2)
