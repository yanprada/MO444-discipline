# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
import random
import game
import util

class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)



class DumbAgent(Agent):
    def getAction(self,state):
        "The agent receives a GameState (defined in pacman.py)."
        print("Location: ", state.getPacmanPosition())
        print("Actions available:", state.getLegalPacmanActions())
        random_num = random.randint(0, len(state.getLegalPacmanActions())-1)
        direction = str(state.getLegalPacmanActions()[random_num].upper())
        print(direction)
        if direction == 'WEST':
            return Directions.WEST
        elif direction == 'NORTH':
            return Directions.NORTH
        elif direction == 'EAST':
            return Directions.EAST
        elif direction == 'SOUTH':
            return Directions.SOUTH
        return Directions.STOP

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = []
        for i in range(0,5):
            self.actionList.append(Directions.STOP);
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        populationList = []
        tempState=currstate=state
        currmax = gameEvaluation(currstate,tempState)
        bestaction=self.actionList[0]
        possible = tempState.getAllPossibleActions();
        flag=True
        loseflag=False

        for i in range(0,8):
            for j in range(0, len(self.actionList)):
                self.actionList[j] = possible[random.randint(0, len(possible) - 1)]
            populationList.append(self.actionList[:])

        while flag:
            population = []
            for i in range(0,8):
                tempState=state
                for j in range(0, len(self.actionList)):
                    tempState = tempState.generatePacmanSuccessor(populationList[i][j])
                    if tempState==None:
                        flag=False
                        break
                    if tempState.isLose():
                        break
                    elif tempState.isWin():
                        return self.actionList[0]
                if flag==False:
                    break
                population.append((populationList[i][:],gameEvaluation(currstate,tempState)))

            if flag==False:
                break
            population.sort(key = lambda x:x[1], reverse=True)
            if population[0][1]>currmax:
                bestaction=population[0][0][0]

            populationList=[]
            for i in range (0,4):
                rankpart1,rankpart2 = self.pairSelect()
                x_chromosome = population[rankpart1][0][:]
                y_chromosome = population[rankpart2][0][:]
                crossover = random.randint(0,100)

                if random.randint(0,100)<=70:
                    child1=[]
                    child2=[]
                    for j in range(0,5):
                        if random.randint(0,1)==1:
                            child1.append(x_chromosome[j])
                            child2.append(y_chromosome[j])
                        else:
                            child1.append(y_chromosome[j])
                            child2.append(x_chromosome[j])
                    populationList.append(child1)
                    populationList.append(child2)
                else:
                    populationList.append(x_chromosome[:])
                    populationList.append(y_chromosome[:])

            for i in range(0,8):
                if random.randint(0, 100) <= 10:
                    populationList[i][random.randint(0,4)] = possible[random.randint(0,len(possible)-1)]

        return bestaction

class GeneticAgent2(Agent):
	# Initialization Function: Called one time when the game starts
	def registerInitialState(self, state):
		return;

	# GetAction Function: Called with every frame
	def getAction(self, state):

		# TODO: write Genetic Algorithm instead of returning
		possible = state.getAllPossibleActions()

		def init_population():
			gene_pool = state.getAllPossibleActions()
			g_len = len(gene_pool)
			population = []
			for i in range(0, 8):
				chromosome = []
				for j in range(0, 5):
					chromosome.append(
						gene_pool[
							random.randint(0, g_len - 1)]
					)
				population.append(chromosome)
			return population

		def selection_chances(state, population):
			chromosome_score = []
			score = 0
			tempstate = state
			result = []
			for direction_list in population:
				score = 0
				flag = True
				tempstate = state
				for direction in direction_list:
					if tempstate.isWin() + tempstate.isLose() == 0:
						successor = \
							tempstate.generatePacmanSuccessor(
							direction
						)
						if successor == None:
							flag = False
							break;
						tempstate = successor
						score = score + scoreEvaluation(successor)
					else:
						break
				if flag == False:
					break;
				chromosome_score.append(direction_list)
				chromosome_score_list = (score, direction_list)
				result.append(chromosome_score_list)
			result.sort(reverse=True)
			if flag is False:
				return 0
			else:
				return result

		def rank_selection(sorted_chromosome_list):
			total_chromosomes = 8
			tot_sum = 1
			denominator = sum(range(1, total_chromosomes + 1))
			proportional_probability = [
				round((x + 1) * 1.0 / denominator, 2) for x in
				range(8)]
			proportional_probability.sort(reverse=True)
			x = 0.00
			iterative_proportional_probability = []
			for i in range(len(proportional_probability)):
				iterative_proportional_probability.append(
					x + proportional_probability[i])
				x = iterative_proportional_probability[i]

			limit = random.uniform(0, tot_sum)
			index_of_first_item_bigger_than = next(
				x[0] for x in
				enumerate(iterative_proportional_probability)
				if x[1] > limit
			)
			return \
			sorted_chromosome_list[
				index_of_first_item_bigger_than][1]

		# rank_selection(sorted_chromosome_list)
		def chromosomeCrossover(parent1, parent2):
			new_chromosome = []
			for i in range(0, 5):
				x = random.uniform(0, 1)
				if x < 0.5:
					new_chromosome.append(parent1[i])
				else:
					new_chromosome.append(parent2[i])

			return new_chromosome;

		def chromosomeMutate(new_population):
			for chromosome in new_population:
				r_num = random.randint(0, 1)
				if r_num <= 0.1:
					chromosome[random.randint(0, 4)] = possible[
						random.randint(0, len(possible) - 1)]
			return new_population

		def gen_algo():
			population = init_population()
			population_selection = selection_chances(state,
													 population)
			while True:
				if population_selection == 0:
					break;
				else:
					parent1 = rank_selection(population_selection)
					parent2 = rank_selection(population_selection)
					r_test = random.randint(0, 1)
					if r_test < 0.7:
						for i in range(0, 2):
							new_individual = chromosomeCrossover(
								parent1, parent2)
							for i in range(len(population)):
								if population[i] == parent1 or \
												population[
													i] == parent2:
									population[i] = new_individual
					new_population = chromosomeMutate(population)
					population_selection = selection_chances(state,
															 new_population)

					next_action = population_selection[1][1][0];
					return next_action

		next_action = gen_algo()
		return next_action
class GeneticAgent3(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    def rankSelection(self):
        i = random.randint(1, 36)

        if i == 1:
            return 1
        elif i < 4:
            return 2
        elif i < 7:
            return 3
        elif i < 11:
            return 4
        elif i < 16:
            return 5
        elif i < 22:
            return 6
        elif i < 29:
            return 7
        elif i < 37:
            return 8

    # GetAction Function: Called with every frame
    def getAction(self, state):
        possible = state.getAllPossibleActions()
        # initial populations and its initializations

        population = []
        flag = True
        for i in range(0, 8):
            tempstate = state
            actionList = []
            for j in range(0, 5):
                actionList.append(possible[random.randint(0, len(possible) - 1)])

            for j in range(0, 5):
                tempstate = tempstate.generatePacmanSuccessor(actionList[j])
                if tempstate.isWin():
                    return actionList[0]
                elif tempstate.isLose():
                    break

            tempvalue = gameEvaluation(state, tempstate)

            population.append([actionList, tempvalue])

        while flag:
            population.sort(key=lambda population: population[1])

            temppopulation = []
            # rankselection and crossover
            for q in range(0, 4):

                rank1 = self.rankSelection() - 1
                rank2 = self.rankSelection() - 1
                if random.randint(1, 100) <= 70:
                    tempactionlist1 = []
                    tempactionlist2 = []
                    # TODO pair will generate two children by crossing- over. and save it in tempactionlist
                    for i in range(0, 5):
                        if random.randint(1, 100) < 50:
                            temp = (population[rank1][0])
                            tempactionlist1.append(temp[i])
                        else:
                            temp = (population[rank2][0])
                            tempactionlist1.append(temp[i])

                        if random.randint(1, 100) < 50:
                            temp = (population[rank1][0])
                            tempactionlist2.append(temp[i])
                        else:
                            temp = (population[rank2][0])
                            tempactionlist2.append(temp[i])

                    temppopulation.append(tempactionlist1)
                    temppopulation.append(tempactionlist2)
                else:
                    temppopulation.append(population[rank1][0])
                    temppopulation.append(population[rank2][0])

            # mutation

            for i in range(0, 8):
                if random.randint(1, 100) <= 10:
                    place = random.randint(0, 4)
                    temp = temppopulation[i]
                    temp[place] = possible[random.randint(0, len(possible) - 1)]
                    temppopulation[i] = temp

            # score evaluation
            for i in range(0, 8):
                tempstate = state
                actionList = temppopulation[i]

                for j in range(0, 5):
                    tempstate = tempstate.generatePacmanSuccessor(actionList[j])
                    if tempstate is None:
                        flag = False
                        break
                    elif tempstate.isWin():
                        return actionList[0]
                    elif tempstate.isLose():
                        break

                if flag:
                    tempvalue = gameEvaluation(state, tempstate)
                else:
                    break

                population[i] = [actionList, tempvalue]

        population.sort(key=lambda population: population[1])
        actionListbest = population[7][0]

        return actionListbest[0]
class GeneticAgent4(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    def ranking(self, state,chromosome):
        while len(chromosome)!=0:
            actionList=chromosome.pop(0)
            temp=actionList[:]
            finalstate=state
            flag=0
            while len(actionList)!=0:
                action=actionList.pop(0)
                finalstate=finalstate.generatePacmanSuccessor(action)
                if finalstate is None:
                    flag=1
                    break
                elif finalstate.isWin():
                    break
                elif finalstate.isLose():
                    break
            if flag==1:
                break
            else:
                score=gameEvaluation(state,finalstate)
                self.sequencescore.append((temp,score))
        if flag!=1:
            self.sequencescore=sorted(self.sequencescore,key=lambda x:x[1])
            for i in range(1,len(self.sequencescore)+1):
                self.sequencescore[i-1]=(i,self.sequencescore[i-1][1],self.sequencescore[i-1][0])
            return self.sequencescore,flag
        else:
            return self.sequencescore,1

    def probability(self):
        n=len(self.sequencescore)
        ranksum=float(((n)*(n+1))/2)
        for i in range(0,len(self.sequencescore)):
            self.sequencescore[i]=(self.sequencescore[i][0],self.sequencescore[i][0]/ranksum,self.sequencescore[i][1],self.sequencescore[i][2])
        return self.sequencescore

    def parentSelection(self):
        parentList=[]
        flag=0
        while len(parentList)<2:
            random_num=random.uniform(0,1)
            i=0
            while len(self.sequencescore)!=0 and i<len(self.sequencescore):
                prob=self.sequencescore[i][1]
                if prob>=random_num:
                    parentList.append(self.sequencescore[i][3])
                    self.sequencescore.pop(i)
                    if len(parentList)==2:
                        flag=1
                        break

                i+=1
            if flag==1:
                break
        return parentList

    def crossover(self,first_parent,second_parent):
        cross=[]
        for i in range(0,len(first_parent)):
            r=random.randint(0,10)
            if r<5:
                cross.append(first_parent[i])
            else:
                cross.append(second_parent[i])
        return cross

    def mutation(self,state,new_generation):
        possible=state.getAllPossibleActions()
        for i in range(0, len(new_generation)):
            r = random.randint(0,10)
            if r <= 1:
                rand = random.randint(0,4)
                action = possible[random.randint(0, len(possible) - 1)]
                new_generation[i][rand] = action
        return new_generation


#performs both crossover and mutation
    def cross_mutation(self,state,parentList):
        rand=random.randint(0,10)
        a,b=parentList
        new_generation=[]
        if rand<7:
            new_generation.append(self.crossover(a,b))
            new_generation.append(self.crossover(b,a))
        else:
            new_generation.append(a)
            new_generation.append(b)
        new_chromosomes=self.mutation(state,new_generation)
        return new_chromosomes



    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        possible=state.getAllPossibleActions()
        actionList=range(5)
        chromosome=[]
        for i in range(0, 8):
            actionList = []
            for j in range(0, 5):                           # 5= lenActionList
                actionList.append(possible[random.randint(0, len(possible)-1)])
            chromosome.append(list(actionList))
        self.sequencescore=[]
        max_value=[0,""]
        while(True):
            new_generation=[]
            self.sequencescore,flag=self.ranking(state,chromosome)
            if flag==1:
                break
            #To check value against max_value
            if self.sequencescore[len(self.sequencescore)-1][1]>max_value[0]:
                max_value=(self.sequencescore[len(self.sequencescore)-1][1],self.sequencescore[len(self.sequencescore)-1][2][0])

            self.sequencescore=self.probability()
            while(len(self.sequencescore)!=0):
                parentList=self.parentSelection()
                cross_mutationList=self.cross_mutation(state,parentList)
                for i in cross_mutationList:
                    new_generation.append(i)

            chromosome=new_generation[:]

        return max_value[1]

class GeneticAgent5(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    def ranking(self, state,chromosome):
        while len(chromosome)!=0:
            actionList=chromosome.pop(0)
            temp=actionList[:]
            finalstate=state
            flag=0
            while len(actionList)!=0:
                action=actionList.pop(0)
                finalstate=finalstate.generatePacmanSuccessor(action)
                if finalstate is None:
                    flag=1
                    break
                elif finalstate.isWin():
                    break
                elif finalstate.isLose():
                    break
            if flag==1:
                break
            else:
                score=gameEvaluation(state,finalstate)
                self.sequencescore.append((temp,score))
        if flag!=1:
            self.sequencescore=sorted(self.sequencescore,key=lambda x:x[1])
            for i in range(1,len(self.sequencescore)+1):
                self.sequencescore[i-1]=(i,self.sequencescore[i-1][1],self.sequencescore[i-1][0])
            return self.sequencescore,flag
        else:
            return self.sequencescore,1

    def probability(self):
        n=len(self.sequencescore)
        ranksum=float(((n)*(n+1))/2)
        for i in range(0,len(self.sequencescore)):
            self.sequencescore[i]=(self.sequencescore[i][0],self.sequencescore[i][0]/ranksum,self.sequencescore[i][1],self.sequencescore[i][2])
        return self.sequencescore

    def parentSelection(self):
        parentList=[]
        flag=0
        while len(parentList)<2:
            random_num=random.uniform(0,1)
            i=0
            while len(self.sequencescore)!=0 and i<len(self.sequencescore):
                prob=self.sequencescore[i][1]
                if prob>=random_num:
                    parentList.append(self.sequencescore[i][3])
                    self.sequencescore.pop(i)
                    if len(parentList)==2:
                        flag=1
                        break

                i+=1
            if flag==1:
                break
        return parentList

    def crossover(self,first_parent,second_parent):
        cross=[]
        for i in range(0,len(first_parent)):
            r=random.randint(0,10)
            if r<5:
                cross.append(first_parent[i])
            else:
                cross.append(second_parent[i])
        return cross

    def mutation(self,state,new_generation):
        possible=state.getAllPossibleActions()
        for i in range(0, len(new_generation)):
            r = random.randint(0,10)
            if r <= 1:
                rand = random.randint(0,4)
                action = possible[random.randint(0, len(possible) - 1)]
                new_generation[i][rand] = action
        return new_generation


#performs both crossover and mutation
    def cross_mutation(self,state,parentList):
        rand=random.randint(0,10)
        a,b=parentList
        new_generation=[]
        if rand<7:
            new_generation.append(self.crossover(a,b))
            new_generation.append(self.crossover(b,a))
        else:
            new_generation.append(a)
            new_generation.append(b)
        new_chromosomes=self.mutation(state,new_generation)
        return new_chromosomes



    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        possible=state.getAllPossibleActions()
        actionList=range(5)
        chromosome=[]
        for i in range(0, 8):
            actionList = []
            for j in range(0, 5):                           # 5= lenActionList
                actionList.append(possible[random.randint(0, len(possible)-1)])
            chromosome.append(list(actionList))
        self.sequencescore=[]
        max_value=[0,""]
        while(True):
            new_generation=[]
            self.sequencescore,flag=self.ranking(state,chromosome)
            if flag==1:
                break
            #To check value against max_value
            if self.sequencescore[len(self.sequencescore)-1][1]>max_value[0]:
                max_value=(self.sequencescore[len(self.sequencescore)-1][1],self.sequencescore[len(self.sequencescore)-1][2][0])

            self.sequencescore=self.probability()
            while(len(self.sequencescore)!=0):
                parentList=self.parentSelection()
                cross_mutationList=self.cross_mutation(state,parentList)
                for i in cross_mutationList:
                    new_generation.append(i)

            chromosome=new_generation[:]

        return max_value[1]
