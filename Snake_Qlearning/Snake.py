import pygame as pg
import Qlearn_Snake
import random
import numpy as np
from sys import exit
import time

class Snake:

	def __init__(self, numRows, numCols):
		# Größe des Spielfeldes
		self.numRows = numRows
		self.numCols = numCols
		self.sHeight, self.sWidth = (self.numRows*15, self.numCols*15)

		# Vorbereitung
		pg.init()
		self.screen = pg.display.set_mode((self.sWidth, self.sHeight))
		pg.display.set_caption("Snake")
		self.clock = pg.time.Clock()

		# Schlange und Nahrung
		self.snake_color = "#8198FE"
		self.dict_directions = {"right":(15,0), "up":(0,-15), "left":(-15,0), "down":(0,15)}
		self.snake_direction = self.dict_directions["right"]
		self.snake = [pg.Rect(1, 1, 13, 13)]
		# Position der Schlangenglieder
		self.head_row = (self.snake[-1].bottom+1) // 15
		self.head_col = (self.snake[-1].right+1) // 15
		self.snake_rects_RowCol = [[self.head_row, self.head_col]]

		# Nahrung
		self.food_color = "#FA9A7C"
		self.food = pg.Rect(3+15*random.randint(1, self.numCols-1), 3+15*random.randint(1, self.numRows-1), 9, 9)
		self.food_row = (self.food.bottom+3) // 15
		self.food_col = (self.food.right+3) // 15

		# für Training
		self.numTimeOuts = 0
		

	def __play(self, player, fps, agent=0):
		""" player = 'human', 'agent' or 'agent train' """

		# Variablen zurücksetzen
		self.snake_direction = self.dict_directions["right"]
		self.snake = [pg.Rect(1, 1, 13, 13)]
		self.food = pg.Rect(3+15*random.randint(1, self.numCols-1), 3+15*random.randint(1, self.numRows-1), 9, 9)
		game_over = False
		food_found = False
		if "train" in player:
			start = time.time()

		while True:
			# alte Richung speichern
			snake_direction_old = self.snake_direction
			# Events checken
			for event in pg.event.get():
				if event.type == pg.QUIT:
					pg.quit()
					exit()
				if player == "human" and event.type == pg.KEYDOWN: 
					if event.key == pg.K_LEFT and snake_direction_old != self.dict_directions["right"]:
						self.snake_direction = self.dict_directions["left"]
					if event.key == pg.K_RIGHT and snake_direction_old != self.dict_directions["left"]:
						self.snake_direction = self.dict_directions["right"]
					if event.key == pg.K_UP and snake_direction_old != self.dict_directions["down"]:
						self.snake_direction = self.dict_directions["up"]
					if event.key == pg.K_DOWN and snake_direction_old != self.dict_directions["up"]:
						self.snake_direction = self.dict_directions["down"]

			# = = = = RL PART PREDICTION
			if "agent" in player:
				# aktuellen Zustand kodieren
				direction_idx = list(self.dict_directions.values()).index(self.snake_direction)
				current_state = agent.get_state( self.head_row, self.head_col, direction_idx, self.snake_rects_RowCol, self.food_row, self.food_col, self.numRows, self.numCols)
				# Agent soll neue Richtung festlegen
				action_idx = agent.predict(current_state, direction_idx)
				self.snake_direction = list(self.dict_directions.values())[(direction_idx-(action_idx-1)) % 4]

			# Schlangenkopf bewegen und Position bestimmen
			self.snake.append(self.snake[-1].copy())
			self.snake[-1].move_ip(self.snake_direction)
			self.head_row = (self.snake[-1].bottom+1) // 15
			self.head_col = (self.snake[-1].right+1) // 15

			# Nahrung gefunden?
			self.food_row = (self.food.bottom+3) // 15
			self.food_col = (self.food.right+3) // 15
			if self.food_row == self.head_row and self.food_col == self.head_col:
				# Position für neue Nahrung finden
				while True:
					self.food_row_new = random.randint(0, self.numRows-1)
					self.food_col_new = random.randint(0, self.numCols-1)			
					if [self.food_row_new+1, self.food_col_new+1] not in self.snake_rects_RowCol:
						break
				self.food = pg.Rect(3+15*self.food_col_new, 3+15*self.food_row_new, 9, 9)
				food_found = True

			# falls kein Nahrung, letztes Glied löschen
			if not food_found:
				self.snake.pop(0)

			# Game over?
			if self.head_row <= 0 or self.head_col <= 0 or self.head_row > self.numRows or self.head_col > self.numCols:
				game_over = True
			self.snake_rects_RowCol = [[self.head_row, self.head_col]]
			for rect in self.snake[:-1]:
				self.rect_row = (rect.bottom+1) // 15
				self.rect_col = (rect.right+1) // 15		
				self.snake_rects_RowCol.append([self.rect_row, self.rect_col])
				if self.rect_row == self.head_row and self.rect_col == self.head_col:
					game_over = True

			# = = = = RL PART TRAINING
			if "train" in player:
				# neuen Zustand bestimmen
				direction_idx = list(self.dict_directions.values()).index(self.snake_direction)
				new_state = agent.get_state( self.head_row, self.head_col, direction_idx, self.snake_rects_RowCol, self.food_row, self.food_col, self.numRows, self.numCols)
				#
				reward = 0
				if food_found:
					reward = 1
				if game_over:
					reward = -1
				agent.update_qTable(current_state, action_idx, new_state, reward)

			# Essen zurücksetzen
			food_found = False

			# Spiel neustarten, falls game over
			if game_over:
				break

			# Fenster grau füllen
			self.screen.fill("#F0F0F0")

			# Schlange zeichnen
			for rect in self.snake:
				pg.draw.rect(self.screen, self.snake_color, rect)

			# Nahrung zeichnen
			pg.draw.rect(self.screen, self.food_color, self.food)

			# Runde beenden, wenn sie zu lange läuft (über 2 Sekunden)
			if "train" in player:
				if time.time()-start > 2:
					print("Zeitüberschreitung")
					game_over = True
					self.numTimeOuts += 1

			# updaten und warten
			pg.display.update()
			if fps != -1:
				self.clock.tick(fps)

	def train_agent(self, agent, path_qTable, numGames, verbose=False):
		self.numTimeOuts = 0
		snake_lengths = []
		for i in range(numGames):
			self.__play(player="agent train", fps=-1, agent=agent)
			snake_lengths.append(len(self.snake))
			if verbose and (i+1)%20==0:
				print("Spiel %4.i beendet mit Schlangenlänge %2.i" % (i+1, len(self.snake)))
			if self.numTimeOuts >= 2:
				break
		if path_qTable != "":
			np.savetxt(path_qTable, agent.qTable)
		print("Performance %2.2f mit alpha=%.3f und gamma=%.3f und numGames=%i und Minimum %i" % (sum(snake_lengths[-100:])/len(snake_lengths[-100:]), agent.alpha, agent.gamma, numGames, min(snake_lengths)))

	def play_human(self):
		self.__play(player="human", fps=5)

	def play_agent(self, agent):
		self.__play(player="agent", fps=5, agent=agent)