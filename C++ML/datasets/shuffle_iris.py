import random
lines = open("iris.csv").readlines()
random.shuffle(lines)
open("iris.csv", "w").writelines(lines)