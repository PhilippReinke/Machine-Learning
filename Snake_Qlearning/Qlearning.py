from collections import deque
import numpy as np
"""
	States are decoded by 11 variables with one or zero. They consist of three parts.s
		danger 		left straight right 	-> all combinations are possible
		direction 	right up left down		-> exactly one is true
		food		right above left below	-> one or two can be true
	For example no danger, snake goes right and food is somehwhere up and right decondes to
		000 0100 0110
	This sums up to 8*4*11 = 352 possible states = number of rows in the q table.

	The snake can take three actions
		left straight right
	that are correspond to 0, 1 or 2.

	The reward is +1 for found food, -1 for game over, 0 otherwise.
"""

class QLearn:
	def __init__(self):
		# learning parameters
		self.alpha = 0.1 	# learning rate
		self.gamma = 0.8	# discount factor

		# states
		self.qStates = []
		self.states_first	= ["000", "100", "010", "001", "110", "101", "011", "111"]
		self.states_second	= ["1000", "0100", "0010", "0001"]
		self.states_third	= ["0000", "1000", "0100", "0010", "0001", "1100", "1010", "1001", "0110", "0101", "0011"]
		
		# form all possible combinations
		for first_ele in self.states_first:
			for second_ele in self.states_second:
				for third_ele in self.states_third:
					self.qStates.append(first_ele+second_ele+third_ele)

		# initialise q table with zeros
		#self.qTable = np.zeros((352, 3))
		self.qTable = np.random.rand(352,3)

	def state_to_idx(self, state):
		try:
			pos_first 	= self.states_first.index(state[:3])
			pos_second 	= self.states_second.index(state[3:7])
			pos_third 	= self.states_third.index(state[7:11])
			return pos_third + 11*pos_second + 11*4*pos_first
		except:
			print("Error: State not found.")
			return 0

	def get_state(self, head_row, head_col, direction_idx, snake_rects_RowCol, food_row, food_col, numRows, numCols):
		# assume motion to the right and roate it afterwards
		state = ""
		### dangers
		danger = deque([0,0,0,0]) # last one is for behind
		if head_row == 1:
			danger[0] = 1
		if head_row == numRows:
			danger[2] = 1
		if head_col == 0:
			danger[3] = 1
		if head_col == numCols:
			danger[1] = 1
		# self eating?
		for ele in snake_rects_RowCol[1:]:
			if ele[0] == head_row and ele[1] == head_col+1:
				danger[1] = 1
			if ele[0] == head_row-1 and ele[1] == head_col:
				danger[0] = 1
			if ele[0] == head_row and ele[1] == head_col-1:
				danger[3] = 1
			if ele[0] == head_row+1 and ele[1] == head_col:
				danger[2] = 1
		danger.rotate(-direction_idx)
		danger.pop()
		state += "".join(str(ele) for ele in danger)
		### direction
		direction = deque([0,0,0,0])
		direction[direction_idx] = 1
		state += "".join(str(ele) for ele in direction)
		### food
		food = deque([0,0,0,0])
		if head_row < food_row:
			food[0] = 1
		if head_row > food_row:
			food[2] = 1
		if head_col < food_col:
			food[1] = 1
		if head_col > food_col:
			food[3] = 1
		food.rotate(-direction_idx)
		state += "".join(str(ele) for ele in food)		
		return state

	def predict(self, state, direction_idx):
		idx = self.state_to_idx(state)
		return np.argmax(self.qTable[idx])

	def update_qTable(self, state, action, next_state, reward):
		idx 	 = self.state_to_idx(state)
		idx_next = self.state_to_idx(next_state)
		self.qTable[idx][action] += self.alpha * ( reward + self.gamma*max(self.qTable[idx_next]) -  self.qTable[idx][action] )