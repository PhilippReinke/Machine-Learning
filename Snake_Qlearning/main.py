import numpy as np
import Qlearn_Snake
import Snake

# num_games, greedy, gamma
train_strategy = [(400, True, 0.2), (400, False, 0.4), (400, True, 0.5), (400, False, 0.7), (500, True, 0.7), (500, False, 0.7), (500, True, 0.7), (500, False, 0.7), (500, True, 0.7), (500, False, 0.7)]
agent_path = "qTables/qTable.txt"

# trainieren
print("_____________________________________________________")
print("Runden  Greedy  Gamma  Durchschnitt  Minimum  Maximum")
for i in range(0, len(train_strategy)):

	# Vorbereitung
	game  = Snake.Snake(10, 10)
	agent = Qlearn_Snake.Qlearn_Snake(alpha=0.2, gamma=train_strategy[i][2])
	if i > 0:
		agent.qTable = np.loadtxt(agent_path)

	# Agent trainieren und Q-Tabelle speichern
	game.train_agent(agent, train_strategy[i][0], greedy=train_strategy[i][1])
	np.savetxt("qTables/qTable.txt", agent.qTable)

	# Performance bewerten
	if not train_strategy[i][1]:
		average, minimum, maximum = Snake.Snake(10, 12).evaluate_performance(agent_path)
		print("%6i  %6s    %.1f  %12i  %7i  %7i" %  (train_strategy[i][0], str(train_strategy[i][1]), train_strategy[i][2], average, minimum, maximum) )
	else:
		print("%6i  %6s    %.1f" %  (train_strategy[i][0], str(train_strategy[i][1]), train_strategy[i][2]) )


### Agent spielen lassen
# agent_path = "qTables/qTable_8000.txt"
# game  = Snake.Snake(12, 10)
# agent = Qlearn_Snake.Qlearn_Snake()
# agent.qTable = np.loadtxt(agent_path)

# game.play_agent(agent)
# ##game.play_human()