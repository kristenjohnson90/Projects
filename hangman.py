# HANGMAN GAME
import random


def checklet(y):
    try:
        if len(y) == 1:
            return True
        else:
            raise ValueError
    except ValueError:
        print("You can only guess one letter at a time.")


words = ('hello', 'goodbye', 'colorado', 'conifer', 'mountain', 'hiking', 'climbing')
word = random.choice(words).upper()

length = len(word)
print('{} characters'.format(length))
hang = ['_'] * length

i = 1
while i < 11:
    print('Guess number {}'.format(i))
    letter = input('Guess a letter...\n').upper()

    if checklet(letter):
        i = i + 1
        for c in word:
            if c == letter:
                loc = word.index(c)
                hang[loc] = c
                z = word.count(c)
                if z > 1:
                    loc = word.index(c, loc + 1)
                    hang[loc] = c
                    break

        x = ''.join(hang)
        print(x)
        if x == word:
            print('You did it!')
            exit()

        guess = input('Would you like to guess the word? (Y/N)\n').upper()
        if guess == 'Y':
            guess_word = input('What is the word?\n').upper()
            if guess_word == word:
                print('You did it!')
                break
            else:
                print('Wrong!')

    if i == 11:
        print("Sorry... the word was {}.".format(word))
