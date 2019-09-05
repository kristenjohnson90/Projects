# RANDOM NUMBER GUESS GAME
import random


def checknum(x):
    try:
        if 0 < int(x) < 51:
            return True
    except ValueError:
        print("That's not an int!")


num = random.randint(1, 50)
# print(num)

guess = 0
while guess != num:
    guess = input("Guess the number between 1 and 50...\n")
    if checknum(guess):
        guess = int(guess)
        if guess == num:
            print("You guessed it!")
        elif guess < num:
            diff = num - guess
            print("You are {} too low.".format(diff))
        elif guess > num:
            diff = guess - num
            print("You are {} too high.".format(diff))
    else:
        continue
