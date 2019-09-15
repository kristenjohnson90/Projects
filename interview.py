# between 1 and 1000000
# digits add up to 42

count = 0
for i in range(1,1000001):
    num = sum(list(map(int, str(i))))
    if num == 42:
        count = count + 1
print(count)