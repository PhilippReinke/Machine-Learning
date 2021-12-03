import pygame as pg
from sys import exit
import random
import Qlearning
import time
import numpy as np
import matplotlib.pyplot as plt

# Breite und Höhe müssen
# Vielfaches von 15 sein
numRows = 10; numCols = 10
sHeight, sWidth = (numRows*15, numCols*15)

# Vorbereitung
pg.init()
screen = pg.display.set_mode((sWidth, sHeight))
pg.display.set_caption("Snake")
clock = pg.time.Clock()

# Schlange und Nahrung
snake_color = "#8198FE"
dict_directions = {"right":(15,0), "up":(0,-15), "left":(-15,0), "down":(0,15)}
snake_direction = dict_directions["right"]
snake = [pg.Rect(1, 1, 13, 13)]
# Position der Schlangenglieder
head_row = (snake[-1].bottom+1) // 15
head_col = (snake[-1].right+1) // 15
snake_rects_RowCol = [[head_row, head_col]]

# Nahrung
food_color = "#FA9A7C"
food = pg.Rect(3+15*random.randint(1, numCols-1), 3+15*random.randint(1, numRows-1), 9, 9)
food_row = (food.bottom+3) // 15
food_col = (food.right+3) // 15

# globale Variablen
game_over = False
food_found = False
numGames = 1
numGamesArray = []

# Q Learning
agent = Qlearning.QLearn()
agent.qTable = np.loadtxt("q_tables/qTable_10000.txt")

# Zeitmessung, um Schleifen zu verhindern
start = time.time()

while True:
	# alte Richung speichern
	snake_direction_old = snake_direction
	# Events checken
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			exit()
		# if event.type == pg.KEYDOWN:
		# 	if event.key == pg.K_LEFT and snake_direction_old != dict_directions["right"]:
		# 		snake_direction = dict_directions["left"]
		# 	if event.key == pg.K_RIGHT and snake_direction_old != dict_directions["left"]:
		# 		snake_direction = dict_directions["right"]
		# 	if event.key == pg.K_UP and snake_direction_old != dict_directions["down"]:
		# 		snake_direction = dict_directions["up"]
		# 	if event.key == pg.K_DOWN and snake_direction_old != dict_directions["up"]:
		# 		snake_direction = dict_directions["down"]

	# = = = = RL PART PREDICTION
	#
	# aktuellen Zustand kodieren
	direction_idx = list(dict_directions.values()).index(snake_direction)
	current_state = agent.get_state( head_row, head_col, direction_idx, snake_rects_RowCol, food_row, food_col, numRows, numCols)
	# Agent soll neue Richtung festlegen
	action_idx = agent.predict(current_state, direction_idx)
	snake_direction = list(dict_directions.values())[(direction_idx-(action_idx-1)) % 4]
	# 
	# = = = =

	# Schlangenkopf bewegen und Position bestimmen
	snake.append(snake[-1].copy())
	snake[-1].move_ip(snake_direction)
	head_row = (snake[-1].bottom+1) // 15
	head_col = (snake[-1].right+1) // 15

	# Nahrung gefunden?
	food_row = (food.bottom+3) // 15
	food_col = (food.right+3) // 15
	if food_row == head_row and food_col == head_col:
		# Position für neue Nahrung finden
		while True:
			food_row_new = random.randint(0, numRows-1)
			food_col_new = random.randint(0, numCols-1)			
			if [food_row_new+1, food_col_new+1] not in snake_rects_RowCol:
				break
		food = pg.Rect(3+15*food_col_new, 3+15*food_row_new, 9, 9)
		food_found = True

	# falls kein Nahrung, letztes Glied löschen
	if not food_found:
		snake.pop(0)

	# Game over?
	if head_row <= 0 or head_col <= 0 or head_row > numRows or head_col > numCols:
		game_over = True
	snake_rects_RowCol = [[head_row, head_col]]
	for rect in snake[:-1]:
		rect_row = (rect.bottom+1) // 15
		rect_col = (rect.right+1) // 15		
		snake_rects_RowCol.append([rect_row, rect_col])
		if rect_row == head_row and rect_col == head_col:
			game_over = True

	# = = = = RL PART TRAINING
	# 
	# neuen Zustand bestimmen
	# direction_idx = list(dict_directions.values()).index(snake_direction)
	# new_state = agent.get_state( head_row, head_col, direction_idx, snake_rects_RowCol, food_row, food_col, numRows, numCols)
	# #
	# reward = 0
	# if food_found:
	# 	reward = 1
	# if game_over:
	# 	reward = -1
	# agent.update_qTable(current_state, action_idx, new_state, reward)
	#
	# = = = =

	# Essen zurücksetzen
	food_found = False

	# Spiel neustarten, falls game over
	if game_over:
		#
		numGamesArray.append(len(snake))
		# Nachricht ausgeben
		print("Spiel %3.i beendet mit Schlangenlänge %3.i" % (numGames, len(snake)))
		numGames+=1
		# Variablen zurücksetzen
		snake_direction = dict_directions["right"]
		snake = [pg.Rect(1, 1, 13, 13)]
		food = pg.Rect(3+15*random.randint(1, numCols-1), 3+15*random.randint(1, numRows-1), 9, 9)
		game_over = False
		start = time.time()

	# Fenster grau füllen
	screen.fill("#F0F0F0")

	# Schlange zeichnen
	for rect in snake:
		pg.draw.rect(screen, snake_color, rect)

	# Nahrung zeichnen
	pg.draw.rect(screen, food_color, food)

	# Runde beenden, wenn sie zu lange läuft
	# if time.time()-start > 5:
	# 	game_over = True
	# 	print("Zeitüberschreitung")
	if numGames > 10:
		break

	# updaten und warten
	pg.display.update()
	clock.tick(5) # fps

# sichere die Q-Tabelle
#np.savetxt("q_tables/qTable_10000.txt", agent.qTable)

#
plt.plot(numGamesArray)
plt.show()