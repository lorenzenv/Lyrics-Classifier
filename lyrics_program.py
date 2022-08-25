'''Lyrics program, Spiced 2022, Valentin Lorenzen // Lady Gaga and Thundercat'''

import csv
import pickle 

from art import tprint

print ("\n\n\n" + '\033[94m')
tprint('LYRICS')

# open the clean_corpus
with open('/home/valentin/random-rose-student-code/week_04/clean_corpus.csv', newline='') as file:
    reader = csv.reader(file)
    clean_corpus = list(file)

# open model
with open("/home/valentin/random-rose-student-code/week_04/naive_classifier.bin", "rb") as file:
    model = pickle.load(file)

# function giving artist and probability
def give_artist(lyric):
    print ('\033[96m')
    if not lyric:
        return ("\n\nYou did not give an input.\n\n" + '\033[94m')
    probab = model.predict_proba([lyric])
    who_wrote = model.predict([lyric])[0]
    if round(100*probab[0][0]) == 50:
        return ("\n\nI am sorry, the model cannot decide, try some other lyrics.\n\n" + '\033[94m')
    func_return = "\n'" +  lyric + "' was probably written by:\n--- " + who_wrote
    if who_wrote == "Lady Gaga":
        func_return = func_return + "\nWith a certainty of: " + str(round(100*probab[0][0])) + "%\n"
    else:
        func_return = func_return + "\nWith a certainty of: " + str(round(100*probab[0][1])) + "%\n"
    return (func_return + '\033[94m')

# welcome message
print ('\033[94m' + "Hi.\n\nThis is a little program which takes some lyric as an\ninput and predicts whether that lyric was written by\n- Lady Gaga or\n- Thundercat\n\nType", '\033[1m', "exit" + '\033[0m' + '\033[94m'  + "  to exit\n\n")

# get user input, run function and give output
while True:
    name = input("Enter your lyric: " + '\033[95m')
    if name == "exit":
        print ("\nExiting...\n")
        break
    print(give_artist(name))