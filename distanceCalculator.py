import random
from scipy.spatial import distance


def R(num):
    return round(num)


def genRN(num):
    return 3 * (num - 255) / 19 + random.randrange(-15, 15) + 255


def genRNu255(num):
    return num + random.randrange(-5, 5)


def genRNside(num):
    return R(genRN(num) / random.randrange(2, 3)) + random.randrange(-25, 25)


def genRNsideu255(num):
    return R(genRNu255(num) / random.randrange(2, 3)) + random.randrange(-10, 10)


def genFakeData(num):
    if num > 255:
        return [R(genRN(num)), genRNside(num), genRNside(num), genRNside(num), genRNside(num)]
    else:
        return [R(genRNu255(num)), genRNsideu255(num), genRNsideu255(num), genRNsideu255(num), genRNsideu255(num)]


open_array = []
open_data_array = []
for i in range(1, 31):
    open = random.randrange(2, 1600)
    open_array.append(open)
    open_data_set = genFakeData(open)
    open_data_array.append(open_data_set)
    print(open, open_data_set)

fake_ans = random.randrange(2, 1600)
fake_data_set = genFakeData(fake_ans)
print("*****")
print(fake_ans, fake_data_set)

eucl_dis_array = []
for j in range(0, len(open_data_array)):
    eud = distance.euclidean(open_data_array[j], fake_data_set)
    eucl_dis_array.append(eud)
    # print(eud)

cindex = eucl_dis_array.index(min(eucl_dis_array))

print("*****")
print(open_array)
print("*****")

print(fake_ans)
print(open_array[cindex])