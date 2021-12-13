import unittest
import Qlearn_Snake
import Snake

class TestQlearn_Snake(unittest.TestCase):

	def setUp(self): # before each test
		self.game  = Snake.Snake(10, 15)
		self.agent = Qlearn_Snake.Qlearn_Snake()

	def test_getState(self):
		"""
		recall
			danger 		left straight right 	-> all combinations are possible
			direction 	right up left down		-> exactly one is true
			food		right above left below	-> one or two can be true
		"""

		# test danger boundary and direction of snake
		self.game.snake = [[0,0]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["right"]
		self.assertEqual(self.agent.get_state(self.game), "100 1000 1000".replace(" ", ""))

		self.game.snake = [[0,0]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["left"]
		self.assertEqual(self.agent.get_state(self.game), "011 0010 1000".replace(" ", ""))

		self.game.snake = [[0,0]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["up"]
		self.assertEqual(self.agent.get_state(self.game), "110 0100 1000".replace(" ", ""))

		self.game.snake = [[2,0]]
		self.game.food  = [0,5]
		self.game.snake_direction = self.game.dict_directions["down"]
		self.assertEqual(self.agent.get_state(self.game), "001 0001 1100".replace(" ", ""))

		# other boundary
		self.game.snake = [[9,14]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["right"]
		self.assertEqual(self.agent.get_state(self.game), "011 1000 0110".replace(" ", ""))

		self.game.snake = [[9,0]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["up"]
		self.assertEqual(self.agent.get_state(self.game), "100 0100 1100".replace(" ", ""))

		# test danger of self-eating
		self.game.snake = [[4,5],[2,5],[3,5]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["right"]
		self.assertEqual(self.agent.get_state(self.game), "101 1000 0110".replace(" ", ""))

		self.game.snake = [[4,5],[2,5],[3,5]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["left"]
		self.assertEqual(self.agent.get_state(self.game), "101 0010 0110".replace(" ", ""))

		self.game.snake = [[4,5],[2,5],[3,5]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["up"]
		self.assertEqual(self.agent.get_state(self.game), "010 0100 0110".replace(" ", ""))

		self.game.snake = [[4,5],[2,5],[3,5]]
		self.game.food  = [0,1]
		self.game.snake_direction = self.game.dict_directions["down"]
		self.assertEqual(self.agent.get_state(self.game), "010 0001 0110".replace(" ", ""))


if __name__ == "__main__":
	unittest.main()