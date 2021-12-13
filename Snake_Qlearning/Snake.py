from sys import exit
import pygame as pg
import numpy as np
import random
import time
import os
import Qlearn_Snake
"""
	self.numRows, self.numCols beschreiben die Dimension des Spielfeldes in Anzahl von Quadraten, wobei
	zum Beispiel (0,0) das obere linke Quadrat ist und (0,1) das Quadrat rechts davon. 
	Die Quadrate haben in der Visualisierung eine Seitenlänge 15px.

	self.snake enthält die Koordinaten der Schlangenglieder und self.food die Koordinaten der Nahrung.
"""

class Snake:

	def __init__(self, numRows, numCols):
		# Größe des Spielfeldes
		self.numRows  = max(numRows, 2)
		self.numCols  = max(numCols, 2)

		# Position der Schlange und Nahrung
		self.snake = [np.array([0,0])] # letzter Eintrag ist Kopf
		self.food  = np.array([1,1])

		# Bewegungsrichtung der Schlange
		self.dict_directions = {"right":0, "up":1, "left":2, "down":3}
		self.snake_direction = self.dict_directions["right"]

		# Farben
		self.snake_color = "#8198FE"
		self.food_color  = "#FA9A7C"

	@property
	def snake_rects(self):
		rects = []
		for (row, col) in self.snake:
			rects.append( pg.Rect(1+15*col, 1+15*row, 13, 13) )
		return rects

	@property
	def food_rect(self):
		(row, col) = self.food
		return pg.Rect(3+15*col, 3+15*row, 9, 9)

	@property 
	def snake_direction_coordinates(self):
		return  np.array([[0,1], [-1,0], [0,-1], [1,0]][self.snake_direction])

	def __play(self, player="human", fps=5, agent=0):
		# PyGame vorbereiten
		pg.init()
		screen = pg.display.set_mode( (self.numCols*15, self.numRows*15) )
		pg.display.set_caption("Snake")
		clock = pg.time.Clock()

		# Startbedingungen herstellen
		self.snake = [np.array([2,2])]
		self.food = np.array([random.randint(0, self.numRows-1), random.randint(1, self.numCols-1)])
		self.snake_direction = 0
		found_food = False
		game_over = False

		# Loops im Training stoppen
		if "train" in player:
			start = time.time()

		while not game_over:
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
				# gegenwärtigen Zustand speichern für Training
				current_state = agent.get_state(self)
				# Agent soll neue Richtung festlegen
				action, self.snake_direction = agent.predict(self)

			# Schlange bewegen, aber letzes Glied noch nicht löschen
			self.snake.append( self.snake[-1].copy() )
			self.snake[-1] += self.snake_direction_coordinates

			# Nahrung gefunden?
			if np.all(self.food == self.snake[-1]):
				found_food = True
				# Nahrung neu positionieren
				while np.any(np.all(self.food == self.snake, axis=1)):
					self.food = np.array([random.randint(0, self.numRows-1), random.randint(0, self.numCols-1)])

			# Kopf beißt Schlange?
			if np.any(np.all(self.snake[-1] == self.snake[:-1], axis=1)):
				game_over = True
			# Spielfeld verlassen?
			if np.any(self.snake[-1] < np.array([0,0])) or np.any(self.snake[-1] >= np.array([self.numRows,self.numCols])):
				game_over = True

			# Runde beenden, wenn sie zu lange läuft (über 2 Sekunden)
			if "train" in player:
				if time.time()-start > 5:
					print("Zeitüberschreitung")
					agent.qTable[agent.state_to_idx(current_state)][action] = 0
					start = time.time()

			# = = = = RL PART TRAINING
			if "train" in player:
				# neuen Zustand bestimmen
				new_state = agent.get_state(self)
				# Belohnung bestimmen
				reward = 0
				if found_food:
					reward = 1
				if game_over:
					reward = -1
				# q-Tabelle updaten
				agent.update_qTable(current_state, action, new_state, reward)

			# ggf letztes Glied löschen und Variable zurücksetzen
			if not found_food:
				self.snake.pop(0)
			found_food = False

			# zeichne Hintergrund, Schlange und Nahrung
			screen.fill("#F0F0F0")
			for rect in self.snake_rects:
				pg.draw.rect(screen, self.snake_color, rect)
			pg.draw.rect(screen, self.food_color, self.food_rect)

			# updaten und warten
			pg.display.update()
			if fps != -1:
				clock.tick(fps)

		# PyGame beenden
		pg.quit()

	def play_human(self):
		self.__play(player="human", fps=5)

	def play_agent(self, agent):
		os.environ["SDL_VIDEODRIVER"] = "dummy"
		self.__play(player="agent", fps=-1, agent=agent)

	def train_agent(self, agent, numGames):
		# Ausgabe nicht anzeigen
		os.environ["SDL_VIDEODRIVER"] = "dummy"
		# trainieren
		try:
			print("Runde  Länge")
			for i in range(1, numGames+1):
				self.__play(player="agent train", fps=-1, agent=agent)
				if i%20 == 0:
					print("%5i %5i" % (i, len(self.snake)))
		finally:
			# Ausgabemodus zurücksetzen
			os.environ["SDL_VIDEODRIVER"] = ""