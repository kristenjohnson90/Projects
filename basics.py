fruit = "banana"
print(fruit[:3])
print(fruit[3:])

myList = ["Kristen", "Ryan", "Sydney"]

nums = list(range(5))
accum = []
for i in nums:
    x = i**2
    accum.append(x)
print(accum)

del accum[0]
accum.remove(16)

fname = 'text.txt'
with open(fname, 'w') as md:
    for i in myList:
        md.write("My name is: {}\n".format(i))
    md.write(str(accum))

with open(fname, 'r') as md:
    print(md.read())

# DICTIONARY
pets = {'dog':'Taz','cat':'Kimba'}
print(pets)
print(pets['dog'])
print(list(pets.keys()))


# FUNCTIONS
def hello(y, name):
    a = "Hello, nice to meet you {}.".format(name)
    z = y**2
    return z, a


result = hello(2, "Kristen")
print("Square is {} and message is {}".format(result[0], result[1]))

theSum = 0
x = 0
while (x != 0):
    x = int(input("next number..."))
    theSum = theSum + x

print(theSum)


L = ['A', 'B', 'I', 'I','A','Z','I']

d={}
for x in L:
    if x in d:
        d[x] = d[x] + 1
    else:
        d[x] = 1
for x in sorted(d.keys(), key=lambda k: d[k], reverse=True):
    print("{} appears {} times".format(x, d[x]))


nested = [[1, 2, 3],[4, 5], ["blue", ["burgundy", "maroon"]]]
print(nested, nested[0], nested[0][2])
# burgundy
print(nested[2][1][0])

# nested dictionary
info = {'personal_data':
            {'name':'Kristen',
             'age':29,
             'major':'Atmospheric Science',
             'physical_features':
                 {'color':{'eye':'blue',
                           'hair':'brown'},
                  'height':"5'3"
                  }
             },
        'other':
            {'favorite_colors':['turquoise','pink']
             }
        }

#blue
eye_color = info['personal_data']['physical_features']['color']['eye']
print(eye_color)

#brown
info['personal_data']['physical_features']['color']['eye'] = 'brown'
print(info['personal_data']['physical_features']['color']['eye'])


# nested iteration
print("Start nested loop...")
ind = []
for x in nested:
    print("level1: {}".format(x))
    for y in x:
        if type(y) is list:
            for z in y:
                print("         level3: {}".format(z))
                ind.append(z)
        else:
            print("     level2: {}".format(y))
            ind.append(y)
print(ind)




