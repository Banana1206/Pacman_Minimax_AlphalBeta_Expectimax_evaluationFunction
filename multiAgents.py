# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# link: https://github.com/brandhaug/pacman-multiagent/blob/master/multiAgents.py
# link: https://github.com/RobinManhas/Pacman-Minimax-Alpha-Beta-Expectimax/blob/master/multiAgents.py


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

queue = []
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # score = 0
        # # if state is win
        # if successorGameState.isWin():
        #     return 10000
        #
        # # if pacman eats food
        # if (currentGameState.getFood()[newPos[0]][newPos[1]]):
        #     score += 20
        #
        # # find closest food
        # closest_food = min([manhattanDistance(newPos, food) for food in newFood.asList()])
        #
        # # find out policy by state of ghost
        # for ghost in newGhostStates:
        #     if ghost.scaredTimer > 0:
        #         if manhattanDistance(ghost.getPosition(), newPos)==0:
        #             score += ghost.scaredTimer*10
        #         else:
        #             score += ghost.scaredTimer*2
        #     else:
        #         if manhattanDistance(ghost.getPosition(), newPos) <= 2:
        #             score -= 100
        #         else:
        #             if closest_food < manhattanDistance(ghost.getPosition(), newPos):
        #                 score += 10
        #             else:
        #                 score += 5
        # # if pacman moves away food, minus score by distance of closest food
        # score -= closest_food
        #
        # #if pacman is stop, minus score
        # if action == 'Stop':
        #     score -= 50
        #
        #
        # queue.insert(0, newPos)
        # if len(queue)>3:
        #     queue.pop()
        #     if queue[0] == queue[2] or queue[0] == queue[1]:
        #         score -= 1
        #
        # print("queue: " ,queue)
        # print("total score: ",score)
        # return score

        # ===========================================================================

        # #return successorGameState.getScore()
        # food = currentGameState.getFood()
        # currentPos = list(successorGameState.getPacmanPosition())
        # distance = float("-Inf")
        #
        # foodList = food.asList()
        #
        # if action == 'Stop':
        #     return float("-Inf")
        #
        # for state in newGhostStates:
        #     if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
        #         return float("-Inf")
        #
        # for x in foodList:
        #     tempDistance = -1 * (manhattanDistance(currentPos, x))
        #     if (tempDistance > distance):
        #         distance = tempDistance
        #
        # return distance


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

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """



    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # pacmanIndex = 0
        # return self.performMinimax(1, pacmanIndex, gameState)
        #util.raiseNotDefined()

        def alphabeta(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        # action = alphabeta(gameState)
        #
        # return action
#=================================================================
        # def Minimax_search(depth, agentIndex, gameState):
        #     if(gameState.isWin() or gameState.isLose() or depth> self.depth):
        #         return self.evaluationFunction(gameState)
        #
        #
        #
        # return Minimax_search(1,0,gameState)
#=================================================================
        def minimax_search(state, agentIndex, depth):
            # if in min layer and last ghost
            if agentIndex == state.getNumAgents():
                # if reached max depth, evaluate state
                if depth == self.depth:
                    return self.evaluationFunction(state)
                # otherwise start new max layer with bigger depth
                else:
                    return minimax_search(state, 0, depth + 1)
            # if not min layer and last ghost
            else:
                moves = state.getLegalActions(agentIndex)
                # if nothing can be done, evaluate the state
                if len(moves) == 0:
                    return self.evaluationFunction(state)
                # get all the minimax values for the next layer with each node being a possible state after a move
                next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

                # if max layer, return max of layer below
                if agentIndex == 0:
                    return max(next)
                # if min layer, return min of layer below
                else:
                    return min(next)
        # select the action with the greatest minimax value
        result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))
        return result

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    # def Minimax_AlphaBeta(self, depth, agentIndex, gameState, alpha, beta):
    #
    #     if (gameState.isWin() or gameState.isLose() or depth > self.depth):
    #         return self.evaluationFunction(gameState)
    #
    #     list_value = [] # stores value for each node action
    #     list_action = gameState.getLegalActions(agentIndex) # store action
    #     if Directions.STOP in list_action:
    #         list_action.remove(Directions.STOP)
    #
    #     for action in list_action:
    #         successor = gameState.generateSuccessor(agentIndex, action)
    #
    #         if(agentIndex+1) >= gameState.getNumAgents():
    #             value = self.Minimax_AlphaBeta(depth+1, 0, successor, alpha, beta)
    #         else:
    #             value = self.Minimax_AlphaBeta(depth, agentIndex+1, successor, alpha, beta)
    #
    #         if (agentIndex == 0 and value > alpha):
    #             alpha = value
    #
    #         if (agentIndex > 0 and value < beta):
    #             beta = value
    #         list_value += [value]
    #
    #     if agentIndex ==0:
    #         if depth==1:
    #             max_score = max(list_value)
    #             lenght = len(list_value)
    #
    #             for i in lenght:
    #                 if list_value[i] == max_score:
    #                     return list_action[i]
    #         else:
    #             ret_val = max(list_value)
    #     elif agentIndex >0:
    #         reval = min(list_value)
    #     return reval

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')

        action_value = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            action_value = self.Min_value(gameState.generateSuccessor(0, action),1, 0 , alpha, beta)
            if (alpha < action_value):
                alpha = action_value
                best_action = action
        return best_action


    def Min_value(self, gameState, agentIndex, depth, alpha, beta):

        if (len(gameState.getLegalActions(agentIndex))==0): #No legal actions
            return self.evaluationFunction(gameState)

        action_value = float('inf') #init maximum value action
        for action in gameState.getLegalActions(agentIndex):
            if(agentIndex < gameState.getNumAgents()-1):
                action_value = min(action_value, self.Min_value(gameState.generateSuccessor(agentIndex, action)
                                                                , agentIndex+1,depth, alpha, beta))
            else: # the last ghost HERE
                action_value = min(action_value, self.Max_value(gameState.generateSuccessor(agentIndex, action)
                                                                , depth + 1, alpha, beta))
            if action_value < alpha: #pruning branch
                return action_value
            beta= min(beta, action_value)

        return action_value

    def Max_value(self, gameState, depth, alpha, beta):
        # Return max agents best move

        if depth==self.depth or len(gameState.getLegalActions(0))==0:
            return  self.evaluationFunction(gameState)
        action_value = float('-inf')

        for action in gameState.getLegalActions(0):
            action_value = max(action_value, self.Min_value(gameState.generateSuccessor(0,action)
                               , 1, depth, alpha, beta ))
            if action_value >  beta:
                return action_value
            alpha = max(alpha, action_value)
        return action_value




        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacman_legal_actions = gameState.getLegalActions(0)  # all the legal actions of pacman.
        max_value = float('-inf')
        max_action = None  # one to be returned at the end.

        for action in pacman_legal_actions:  # get the max value from all of it's successors.
            action_value = self.Min_Value(gameState.generateSuccessor(0, action), 1, 0)
            if ((action_value) > max_value):  # take the max of all the children.
                max_value = action_value
                max_action = action

        return max_action  # Returns the final action .
        # util.raiseNotDefined()

    def Max_Value(self, gameState, depth):
        """For the Max Player here Pacman"""

        if ((depth == self.depth) or (len(gameState.getLegalActions(0)) == 0)):
            return self.evaluationFunction(gameState)

        return max([self.Min_Value(gameState.generateSuccessor(0, action), 1, depth) for action in
                    gameState.getLegalActions(0)])

    def Min_Value(self, gameState, agentIndex, depth):
        """ For the MIN Players or Agents  """

        num_actions = len(gameState.getLegalActions(agentIndex))

        if (num_actions == 0):  # No Legal actions.
            return self.evaluationFunction(gameState)

        if (agentIndex < gameState.getNumAgents() - 1):
            return sum(
                [self.Min_Value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in
                 gameState.getLegalActions(agentIndex)]) / float(num_actions)

        else:  # the last ghost HERE
            return sum([self.Max_Value(gameState.generateSuccessor(agentIndex, action), depth + 1) for action in
                        gameState.getLegalActions(agentIndex)]) / float(num_actions)

    # def Max_value(self, gameState, depth):
    #
    #     if ((depth == self.depth) or (len(gameState.getLegalActions(0)) == 0)):
    #         return self.evaluationFunction(gameState)
    #
    #     return  max([self.Min_value(gameState.generateSuccessor(0, action),
    #                                 1, depth) for action in gameState.getLegalActions(0)])
    # def Min_value(self, gameState, agentIndex, depth):
    #
    #     num_action = len(gameState.getLegalActions(agentIndex))
    #     if(num_action == 0): #no legal action
    #         return self.evaluationFunction(gameState)
    #     if(agentIndex < gameState.getNumAgents()-1):
    #         return  sum([self.Min_value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
    #                      for action in gameState.getLegalActions(agentIndex)]) / float(num_action)
    #     else: # the last ghost HERE
    #         return sum([self.Max_value(gameState.generateSuccessor(agentIndex, action),depth+1)
    #                      for action in gameState.getLegalActions(agentIndex)]) / float(num_action)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    else:
        closestCapsule = 0

    if closestCapsule:
        closest_capsule = -3 / closestCapsule
    else:
        closest_capsule = 100

    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500

    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0

    return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule

# Abbreviation
better = betterEvaluationFunction

def myEvaluationFunction(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    ghost_scare_dis = 0
    ghost_dis = 0
    if len(newFood.asList())==0:
        return 10000

    # check if the ghost is scared or not
    for ghost in newGhostStates:
        if ghost.scaredTimer > 0:
            ghost_scare_dis = -10/(manhattanDistance(ghost.getPosition(), newPos))
            if (manhattanDistance(ghost.getPosition(), newPos))==0:
                return 1000
        else:
            if manhattanDistance(ghost.getPosition(), newPos) <= 2:
                return -10000
            ghost_dis = -2/(manhattanDistance(ghost.getPosition(), newPos))

    # Get distance of closest capsule
    if len(newCapsules)>0:
        closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    else:
        closestCapsule = 0

    # Get distance of closest food
    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0

    # get same action
    queue.insert(0, newPos)
    same_action = 0
    if len(queue)>3:
        queue.pop()
        if queue[0] == queue[1] or queue[0] == queue[2] :
            same_action = 10

    total_score = -2 * closestFood + ghost_scare_dis + ghost_dis - 10 * len(foodList) + closestCapsule - same_action
    total_score += 8 * (currentGameState.getScore())
    # print("queue: " ,queue)
    # print("total score: ",total_score)
    return total_score
