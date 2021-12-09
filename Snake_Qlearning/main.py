import Snake
import Qlearn_Snake
import numpy as np

# prepare game and agent
game = Snake.Snake(10, 10) 

# prepare game and train agent
# for numGames in [500, 2000, 5000]:
# 	print("==== " + str(numGames))
# 	for alpha in [.01, .05, .1, .2, .3, .4]:
# 		for gamma in [.5, .6, .7, .8, .9]:
# 			agent = Qlearn_Snake.Qlearn_Snake(alpha, gamma)
# 			#agent.qTable = np.loadtxt("q_tables/qTable.txt")
# 			game.train_agent(agent=agent, path_qTable="", numGames=numGames, verbose=False)
# 		print()

# let agent perform a game in normal speed
agent = Qlearn_Snake.Qlearn_Snake()
agent.qTable = np.loadtxt("q_tables/qTable.txt")
game.play_agent(agent)