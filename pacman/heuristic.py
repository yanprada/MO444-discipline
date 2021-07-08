def heuristicDistance( xy1, xy2 ):
    "Returns our distance between points xy1 and xy2"
    return xy1[0] - xy2[0] ,  xy1[1] - xy2[1]
# heuristic for selecting the node
def scoreEvaluation(state):
    return state.getScore() + [0,-1000.0][state.isLose()] + [0,1000.0][state.isWin()]

def normalizedScoreEvaluation(rootState, currentState):
    rootEval = scoreEvaluation(rootState);
    currentEval = scoreEvaluation(currentState);
    return (currentEval - rootEval) / 1000.0;

def euclideanDistance(xy1, xy2):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5
