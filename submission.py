from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  #def getAction(self, gameState):
  """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    #def minimax(depth, GameState.getPacmanState, state):
      #Base case
      #if(self.depth == 0):
        #return evaluationFunction()


      #if(gameState.getLegalActions(agentIndex) == 0):
        #TODO: add maximizing code
      #elif(gameState.getLegalActions(agentIndex) >= 1):
  def getAction(self, gameState):

    """self.numAgents = gameState.getNumAgents()
    self.myDepth = 0
    self.action = Directions.STOP # Imported from a class that defines 5 directions

    def miniMax(gameState, index, depth, action):
      maxU = float('-inf')
      legalMoves = gameState.getLegalActions(index)
      for move in legalMoves:
        tempU = maxU
        successor = gameState.generateSuccessor(index, move)
        maxU = minValue(successor, index + 1, depth)
        if maxU > tempU:
          action = move
      return action

    def maxValue(gameState, index, depth):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)

      index %= (self.numAgents - 1)
      maxU = float('-inf')
      legalMoves = gameState.getLegalActions(index)
      for move in legalMoves:
        successor = gameState.generateSuccessor(index, move)
        maxU = max(maxU, minValue(successor, index + 1, depth))
      return maxU

    def minValue(gameState, index, depth):
      if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)

      minU = float('inf')
      legalMoves = gameState.getLegalActions(index)
      if index + 1 == self.numAgents:
        for move in legalMoves:
          successor = gameState.generateSuccessor(index, move)
          # Where depth is increased
          minU = min(minU, maxValue(successor, index, depth + 1))
      else:
        for move in legalMoves:
          successor = gameState.generateSuccessor(index, move)
          minU = min(minU, minValue(successor, index + 1, depth))
      return minU

    return miniMax(gameState, self.index, self.myDepth, self.action)"""

    def minimax(agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
            return self.evaluationFunction(gameState)
        if agent == 0:  # maximize for pacman
            return max(minimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
        else:  # minize for ghosts
            nextAgent = agent + 1  # calculate the next agent and increase depth accordingly.
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            return min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))

        """Performing maximize action for the root node i.e. pacman"""
    maximum = float("-inf")
    action = Directions.WEST
    for agentState in gameState.getLegalActions(0):
        utility = minimax(1, 0, gameState.generateSuccessor(0, agentState))
        if utility > maximum or maximum == float("-inf"):
            maximum = utility
            action = agentState

    return action
   # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    def maximizer(agent, depth, game_state, a, b):
        v = float("-inf")
        for newState in game_state.getLegalActions(agent):
            v = max(v, alphabetaprune(1, depth, game_state.generateSuccessor(agent, newState), a, b))
            if v > b:
                return v
            a = max(a, v)
        return v

    def minimizer(agent, depth, game_state, a, b):  # minimizer function
        v = float("inf")

        next_agent = agent + 1  # calculate the next agent and increase depth accordingly.
        if game_state.getNumAgents() == next_agent:
            next_agent = 0
        if next_agent == 0:
            depth += 1

        for newState in game_state.getLegalActions(agent):
            v = min(v, alphabetaprune(next_agent, depth, game_state.generateSuccessor(agent, newState), a, b))
            if v < a:
                return v
            b = min(b, v)
        return v

    def alphabetaprune(agent, depth, game_state, a, b):
        if game_state.isLose() or game_state.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
            return self.evaluationFunction(game_state)

        if agent == 0:  # maximize for pacman
            return maximizer(agent, depth, game_state, a, b)
        else:  # minimize for ghosts
            return minimizer(agent, depth, game_state, a, b)

        """Performing maximizer function to the root node i.e. pacman using alpha-beta pruning."""
    utility = float("-inf")
    action = Directions.WEST
    alpha = float("-inf")
    beta = float("inf")
    for agentState in gameState.getLegalActions(0):
        ghostValue = alphabetaprune(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
        if ghostValue > utility:
            utility = ghostValue
            action = agentState
        if utility > beta:
            return utility
        alpha = max(alpha, utility)

    return action
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    def expectimax(agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
            return self.evaluationFunction(gameState)
        if agent == 0:  # maximizing for pacman
            return max(expectimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
        else:  # performing expectimax action for ghosts/chance nodes.
            nextAgent = agent + 1  # calculate the next agent and increase depth accordingly.
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))

    maximum = float("-inf")
    action = Directions.WEST
    for agentState in gameState.getLegalActions(0):
        utility = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
        if utility > maximum or maximum == float("-inf"):
            maximum = utility
            action = agentState

    return action
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """

  # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newFoodList = newFood.asList()
  min_food_distance = -1
  for food in newFoodList:
      distance = util.manhattanDistance(newPos, food)
      if min_food_distance >= distance or min_food_distance == -1:
          min_food_distance = distance

  for ghost_state in currentGameState.getGhostPositions():
      distance = util.manhattanDistance(newPos, ghost_state)
      distances_to_ghosts += distance
      if distance <= 1:
          proximity_to_ghosts += 1

  newCapsule = currentGameState.getCapsules()
  numberOfCapsules = len(newCapsule)

  return currentGameState.getScore() + (1 / float(min_food_distance)) - (1 / float(distances_to_ghosts)) - proximity_to_ghosts - numberOfCapsules
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
