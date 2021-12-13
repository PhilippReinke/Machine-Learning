import numpy as np
import Qlearn_Snake
import Snake

# Spiel starten
# game = Snake.Snake(10, 15)
# game.play_human()

# Agent trainieren
# game  = Snake.Snake(10, 10)
# agent = Qlearn_Snake.Qlearn_Snake()
# agent.qTable = np.loadtxt("qTables/qTable.txt")
# game.train_agent(agent, 100)
# np.savetxt("qTables/qTable.txt", agent.qTable)

# check Performance
game  = Snake.Snake(10, 12)
agent = Qlearn_Snake.Qlearn_Snake()
agent.qTable = np.loadtxt("qTables/qTable_8000.txt")
snake_lengths = []
for _ in range(0, 100):
	game.play_agent(agent)
	snake_lengths.append(len(game.snake))
print("Durchschnitt : " + str(sum(snake_lengths)/100) )
print("Minimum      : " + str(min(snake_lengths)) )
print("Maximum      : " + str(max(snake_lengths)) )

"""

= 8000
Durchschnitt : 24.36
Minimum      : 11
Maximum      : 42

"""