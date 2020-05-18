#househunting2.py

import itertools
import numpy as np 
import numpy.random as random
import copy
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import entropy

import sys
import os
import configparser
import itertools

num_active = 50 # number of scouting ants in the colony, active ants have IDs [0,num_active)
num_passive = 50
num_broods = 100 # brood ants have IDs [num_active, num_active+num_broods)
num_ants = num_active + num_passive + num_broods
nest_qualities = [0, 1, 1.1]
lambda_sigmoid = 4
num_nests = len(nest_qualities)

pop_coeff = 0.5
QUORUM_THRE = 0.15
QUORUM_OFFSET = 0
# Variables for transtional probabilities
search_find = 0.01
follow_find = 0.8
lead_forward = 0.7
transport = 0.95

num_rounds = 2000
percent_conv = 0.9
persist_rounds = 200

global ind
global tandem
global transported

class Ant:
	def __init__(self, _x_id=-1, _cur_state=None, _proposed_action=None):
		if _x_id > -1:
			self.cur_state = _cur_state
			self.proposed_action = _proposed_action
			if _cur_state == None:
				self.cur_state = State()
			if _proposed_action == None:
				self.proposed_action = Action()
		else:
			self.cur_state = None
			self._proposed_action = None
		self.nests_visited = [0]
		self.num_transports = 0
		self.num_tandems = 0
		self.num_rev_tandems = 0
		self.x_id = _x_id

class State:
	def __init__(self, _state_name="at_nest", _home_nest=0, _candidate_nest=-1, _transitioned=False):
		self.state_name = _state_name # name of the state_name
		self.home_nest = _home_nest # default to 0
		self.candidate_nest = _candidate_nest 
		self.transitioned = _transitioned
		self.location = _home_nest
		self.phase = "E" # how much ant is committed to the candidate nest
		self.old_candidate_nest = -1 # only used for reject action

class Action:
	def __init__(self, _action_type="none", _receiving=-1):
		self.action_type = _action_type # type of the action
		self.receiving = _receiving # the id of the receiving ant to this action. 

	def clear_action(self):
		self.action_type = "none"
		self.receiving = -1

class Nest:
	def __init__(self, _id=0, _quality=-1, _committed_ants=0, _adult_ants_in_nest=0):
		self.id = _id
		self.quality = _quality
		self.committed_ants = _committed_ants
		self.ants_in_nest = []
		self.adult_ants_in_nest = _adult_ants_in_nest

Nests = {}
# Dictionary x_id : ant. Note: Ants[-1] is a place holder. 
# Ants[-1] has all members = NULL
Ants = {} 

# list of all the actions involving two ants
all_pair_actions = ["carry", "call"] 

# A 2-level table corresponding to the input actions for each phase.
# First level is the current state of receiving ant The value of the
# second level dictionary item is a pair containing the resulting
# state, and a list of phases that allow this input action for the
# receiving ant in the current state

IT = { "at_nest":{"call":("follow", ["E","A","T"]),
	   "carry":("at_nest", ["E","A","C","T"])}, 
	   "search":{"carry":("at_nest", ["E","A","C","T"])}
	 }

# Four 2-level table corresponding to the Output Actions for each
# phase.  First level is the current state of initiating ant, and
# second is the proposed action The value of the second level
# dictionary item is the resulting state.
OT_E = {"at_nest":{"search":"search","no_action":"at_nest"}, # phase entry point
		"search":{"find":"arrive", "no_action":"at_nest"}, 
		"follow":{"follow_find":"arrive", "get_lost":"search"},
		"arrive":{"reject":"search","no_reject":"at_nest"}
		}
OT_A = {"at_nest":{"search":"search", "accept":"at_nest"}, # phase entry point
	    "search":{"find":"arrive", "no_action":"at_nest"}, 
	    "follow":{"follow_find":"arrive", "get_lost":"search"},
	    "arrive":{"reject":"search","no_reject":"at_nest"}
	   }
OT_C = {"at_nest":{"search":"search", "recruit":"quorum_sensing"},
		"search":{"find":"arrive", "no_action":"at_nest"}, 
		"lead_forward":{"call":"at_nest", "get_lost":"search"},
		"arrive":{"reject":"search","no_reject":"at_nest"},
		"quorum_sensing":{"quorum_met":"transport","quorum_not_met":"lead_forward"} # phase entry point
	   }
OT_T = {"at_nest":{"search":"search", "recruit":"transport"},
		"search":{"find":"arrive", "no_action":"at_nest"},
		"follow":{"follow_find":"arrive", "get_lost":"search"},
		"transport":{"carry":"reverse_lead", "stop_trans":"search"}, # phase entry point
		"reverse_lead":{"no_action":"at_nest","delay":"reverse_lead"},
		"arrive":{"reject":"transport","no_reject":"at_nest"}
	   }
all_OTs = {"E":OT_E, "A":OT_A, "C":OT_C, "T":OT_T}

def initiate_all_ants():
	# Put all active ants in search state first
	for i in range(num_active):
		Ants[i] = Ant(_x_id=i, _cur_state = State(_state_name="search", _home_nest=0, _candidate_nest=-1, _transitioned=False))
	for i in range(num_active, num_ants):
		Ants[i] = Ant(_x_id=i)
	Ants[-1] = Ant()
	Nests[0].committed_ants = num_ants
	Nests[0].ants_in_nest = [x for x in range(0,num_ants)]
	Nests[0].adult_ants_in_nest = num_active+num_passive

def initiate_all_ants_random_loc():
	for i in range(num_ants):
		rand_loc = random.randint(0,num_nests-1)
		rand_state = State(_home_nest=rand_loc)
		Ants[i] = Ant(_x_id=i, _cur_state=rand_state)
		Nests[rand_loc].committed_ants += 1
		Nests[rand_loc].ants_in_nest += [i]
		if i <= num_active+num_passive:
			Nests[rand_loc].adult_ants_in_nest += 1
	Ants[-1] = Ant()

def initiate_nests():
	for i in range(num_nests):
		Nests[i] = Nest(_id=i, _quality=nest_qualities[i])
	# should have at least 1 home nest and 1 candidate nest
	assert(len(Nests) >= 2)
	# Initiate all ants to be in home nest
	#Nests[0].committed_ants = num_ants
	#Nests[0].ants_in_nest = [x for x in range(0,num_ants)]
	#Nests[0].adult_ants_in_nest = num_active+num_passive


def print_all_ants_states(startid, endid):
	for x_id in range(startid, endid):
			ant = Ants[x_id]
			#print(x_id, ant.cur_state.phase, ant.cur_state.state_name, "home:", ant.cur_state.home_nest, "cand:", ant.cur_state.candidate_nest, "old_cand:", ant.cur_state.old_candidate_nest, "loc:", ant.cur_state.location, ant.proposed_action.action_type, "recv:", ant.proposed_action.receiving, ant.nests_visited, ant.num_transports)

def print_all_nests_info(y, y_ants_in_nest):
	nests_visited = [0]*(num_nests+1)
	num_transports_by_nest_visits = [0]*(num_nests+1)
	for x_id in range(num_active):
		nests_visited[len(Ants[x_id].nests_visited)] += 1
		num_transports_by_nest_visits[len(Ants[x_id].nests_visited)] += Ants[x_id].num_transports
	for nest_id in range(num_nests):
		nest = Nests[nest_id]
		y[nest_id].append(nest.committed_ants)
		y_ants_in_nest[nest_id].append(len(nest.ants_in_nest))
		#print("Nest quality", nest.quality, ", ants committed:", nest.committed_ants, ", ants in nest:", len(nest.ants_in_nest), ", adults in nest:", nest.adult_ants_in_nest)
	#print("# ants seeing 1,2,3... nests", nests_visited[1:], "num_transports by # of nests visited:", num_transports_by_nest_visits)
	

def is_active(x_id):
	return x_id < num_active

def is_passive(x_id):
	return (x_id >= num_active and x_id < num_active + num_passive)

def is_brood(x_id):
	return (x_id >= num_active + num_passive and x_id < num_ants)

def sigmoid(x):
	return 1/(1+np.exp(-x*lambda_sigmoid))

# main simulation function
def execute(plot, run_number, csvfile):
	global ind
	global tandem
	global transported
	x = np.arange(0, num_rounds)
	y = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	y_ants_in_nest = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]	
	
	initiate_nests()
	#initiate_all_ants()
	initiate_all_ants_random_loc()
	#best_nest_id = nest_qualities.index(max(nest_qualities))
	conv_start = [-1]*num_nests
	score = 0
	conv_nest = -1
	conv_nest_quality = -100

	ind = 0
	tandem = 0
	transported = 0
	for round in range(num_rounds):
		#print(round)
		execute_one_round()
		print_all_ants_states(0, 50)
		print_all_nests_info(y, y_ants_in_nest)
		if len(Nests[0].ants_in_nest) == 0:
			score = 1/round
			break
		for nest_id in range(1,num_nests):
			if y_ants_in_nest[nest_id][-1] > percent_conv * num_ants: #if converged
				if conv_start[nest_id] == -1:
					conv_start[nest_id] = round
				else:
					if round - conv_start[nest_id] == persist_rounds:
						score = 1/conv_start[nest_id]
						conv_nest = nest_id
						conv_nest_quality = nest_qualities[conv_nest]
			else:
				if conv_start[nest_id] != -1:
					conv_start[nest_id] = -1
		if score > 0 and not plot:
			break
	if not plot:
		row = '"'+str(nest_qualities)+'", '
		row += str(num_ants)+", "+str(pop_coeff)+", "+str(lambda_sigmoid)+", "+str(QUORUM_THRE)+", "+str(QUORUM_OFFSET)
		row += ", "+str(search_find)+", "+str(follow_find)+", "+str(lead_forward)+", "+str(transport)+", "+str(run_number)+", "+str(score)+", "+str(conv_nest)+", "+str(conv_nest_quality)+"\n"
		csvfile.write(row)

		#plt.figure()
		recruit_acts = [(Ants[i].num_tandems + Ants[i].num_transports + Ants[i].num_rev_tandems) for i in range(num_active)]
		bs = [0,1,6,11,16,21,26,31,36]
		# b = 6
		# while b < max(recruit_acts):
		# 	bs.append(b)
		# 	b += 5

		n,_,_ = plt.hist(recruit_acts, bins=bs)
		# plt.hist([(Ants[i].num_transports) for i in range(num_active)], bins=[0,1,6,11,16,21,25,30], density=True)
		print(max(recruit_acts), n)
		#plt.show()
		all_disc = ind + tandem + transported
		
		return score, n, [1.*ind/all_disc, 1.*tandem/all_disc, 1.*transported/all_disc]
	else:
		directory = str(date.today())+'/'
		plt.figure()
		plt.xlabel("Round")
		plt.ylabel("# Ants In Nest")
		plt.plot(x, np.array(y_ants_in_nest[0]), label="HOME Quality "+str(Nests[0].quality), color='r')
		filename = str(Nests[0].quality)+','
		for i in range(1, num_nests):
			plt.plot(x, np.array(y_ants_in_nest[i]), label=("Nest "+str(i)+" Quality "+str(Nests[i].quality)))
			filename += str(Nests[i].quality)+','
		filename += "pop"+str(num_ants)+",coeff"+str(pop_coeff)+",lambda"+str(lambda_sigmoid)
		filename += ",quorum"+str(QUORUM_THRE)+",qoffset"+str(QUORUM_OFFSET)+",sfind"+str(search_find)+",ffind"
		filename += str(follow_find)+",lead"+str(lead_forward)+",trans"+str(transport)+"_"+str(run_number)+",score"+str(score)+",conv_nest"+str(conv_nest)+",conv_nest_q"+str(conv_nest_quality)
		filename += ".png"
		plt.legend(loc='best')
		plt.savefig(directory+filename)
		plt.close()

		plt.figure()
		plt.xlabel("Round")
		plt.ylabel("Entropy of All Ants")
		filename = "ent_"+filename
		entropies = []
		entropies_committed = []
		non_empty_nests = []
		for rd in range(len(y_ants_in_nest[0])):
			ants_in_nests = [y_ants_in_nest[i][rd] for i in range(num_nests)]
			committed_ants_in_nests = [y[i][rd] for i in range(num_nests)]
			entropies.append(entropy(ants_in_nests))
			entropies_committed.append(entropy(committed_ants_in_nests))
			non_empty_nests.append(len([val for val in committed_ants_in_nests if val > 0.1*num_ants/num_nests]))
		plt.plot(x, np.array(entropies), label="Entropy all")
		plt.plot(x, np.array(entropies_committed), label="Entropy committed")
		plt.plot(x, np.array(non_empty_nests), label="# Non-empty nests", color='r')
		plt.legend(loc='best')
		plt.savefig(directory+filename)
		plt.close()
		return 0
	

# This function executes on round of transitions. One action proposed by each ant
def execute_one_round():
	for x_id in range(num_ants):
		ant = Ants[x_id]
		ant.cur_state.transitioned = False
		if x_id < num_active:
			ant.proposed_action = pick_action_and_receiving(x_id, ant.cur_state)
	perm = random.permutation([i for i in range(num_active)])
	step_in_round = 0
	while step_in_round < num_active:
		# set up arguments for the transition call
		x = Ants[perm[step_in_round]]
		y = Ants[x.proposed_action.receiving]
		# perform state transition
		success, new_state_x, new_state_y = transition(x, y)
		# update the states if transition is successful
		if success:
			x.cur_state = new_state_x
			y.cur_state = new_state_y
		#x.proposed_action.clear_action()

		step_in_round += 1

# This function should have x pick an available action and return this action and also the receiving ant's id.
# Note that receiving ant 
def pick_action_and_receiving(x_id, s):
	a = Action()
	a.receiving = -1
	a.action_type = Dista(x_id,s)
	if a.action_type in all_pair_actions:
		src_nest = s.home_nest
		dest_nest = s.candidate_nest
		if a.action_type == "carry":
			dest_nest = s.home_nest
			src_nest = s.candidate_nest
		if src_nest != dest_nest:
			y = Dy(a.action_type, x_id, src_nest)
			a.receiving = y
	return a

# Input: state of x and state y, and an action type
# Return: success flag, new state of x, and new state of y
def transition(x, y):
	state_x = x.cur_state
	state_y = y.cur_state
	action_type = x.proposed_action.action_type

	if not state_x.transitioned:
		if action_type in all_pair_actions:
			if state_y and not state_y.transitioned:
				if action_type == "call":
						x.num_tandems += 1
				success, new_y_state_name = input_transition(y.x_id, state_y, action_type)
				if success:	
					output_transition(x.x_id, state_x, action_type)
					nnest = state_x.candidate_nest
					if action_type == "carry":
						nnest = state_x.home_nest
						x.num_transports += 1
					#print(y.x_id, state_y.state_name, state_y.location, nnest, action_type)
					adjust_nests(y.x_id, state_y, action_type, new_nest = nnest)
					state_y.state_name = new_y_state_name
					state_y.transitioned = True
					return True, state_x, state_y	
			elif (y.x_id == -1 or (state_x.state_name == "transport" and state_y.state_name == "transport") or (state_x.state_name == "lead_forward" and state_y.state_name == "lead_forward")) and random.random() < 0.1:
					state_x.state_name = "at_nest"
					state_x.phase = 'E'
					adjust_nests(x.x_id, state_x, action_type, state_x.home_nest)
					state_x.transitioned = True
					return False, state_x, state_y
			state_x.transitioned = True
		else:
			output_transition(x.x_id, state_x, action_type)
			return True, state_x, state_y
	return False, state_x, state_y

# This function applies action a to state_name s according to the IT table. Returns success and a new state_name as the result
# Returns a pair (success, new_s). Success is True if the action is in the IT table for s. 
# Return (False, s) otherwise
def input_transition(x_id, s, a):
	state = s.state_name
	if not is_active(x_id):
		assert(a == "carry")
		assert(state == "at_nest")
		assert(s.location == s.home_nest)
		return True, "at_nest"
	if (state in IT) and (a in IT[state]) and (s.phase in IT[state][a][1]):
		return True, IT[state][a][0]
	else:
		return False, state

# This function applies action a to state_name s according to the OT table. Returns the resulting state_name
# Should always be a success once called.
def output_transition(x_id, s, a):
	new_state_name = all_OTs[s.phase][s.state_name][a]
	adjust_nests(x_id, s,a)
	assert(is_active(x_id))
	adjust_phase(s,a)
	s.state_name = copy.deepcopy(new_state_name)
	s.transitioned = True

# This function reads the AT table and picks an available action type based on a probability distibution and the current state_name
# returns that action type
def Dista(x_id, s):
	global ind
	global tandem
	global transported
	probs = [1,0]
	if s.state_name == "search":
		probs = [search_find, 1-search_find]
	elif s.state_name == "follow":
		probs = [follow_find,1-follow_find]
	elif s.state_name == "lead_forward":
		probs = [lead_forward, 1-lead_forward]
	elif s.state_name == "tranport":
		probs = [tranport, 1-transport]
	elif s.state_name == "reverse_lead":
		probs = [follow_find,1-follow_find]
	elif s.state_name == "quorum_sensing":
		assert(s.phase == "C")
		probs = [0.92, 0.08]
		if s.candidate_nest != s.home_nest and Nests[s.candidate_nest].adult_ants_in_nest > (QUORUM_THRE*(num_active+num_passive)+QUORUM_OFFSET):
			probs = [0.08, 0.92]
	elif s.state_name == "arrive":
		if s.candidate_nest not in Ants[x_id].nests_visited:
			Ants[x_id].nests_visited.append(s.candidate_nest)
			if Ants[x_id].proposed_action.action_type == 'follow_find':
				tandem += 1
			elif Ants[x_id].proposed_action.action_type == 'find':
				ind += 1
		if s.candidate_nest == s.home_nest:
			happy_prob = 0
		else:
			candidate_q = Nests[s.candidate_nest].quality
			nest_q = Nests[s.home_nest].quality
			standard_diff_q = (candidate_q - nest_q) / 4
			candidate_pop = len(Nests[s.candidate_nest].ants_in_nest)
			nest_pop = len(Nests[s.home_nest].ants_in_nest)
			standard_diff_pop = (candidate_pop - nest_pop) / num_ants
			happy_prob = sigmoid(standard_diff_q + pop_coeff*standard_diff_pop)
			#print("aaaaaaaaaaa", x_id, s.phase, s.state_name, "home:", s.home_nest, "cand:", s.candidate_nest, "loc:", s.location, "cand_pop:", candidate_pop, "home_pop:", nest_pop, happy_prob)

		probs = [1-happy_prob, happy_prob] # [reject, no_reject] or [search, accept/recruit]
	elif s.state_name == "at_nest":
		if s.phase == "A" and s.candidate_nest == s.home_nest:
			happy_prob = 0
		else:
		 	nest_q = Nests[s.location].quality / 4
		 	nest_pop = len(Nests[s.location].ants_in_nest) / num_ants
		 	happy_prob = sigmoid(nest_q + pop_coeff*nest_pop)
		 	#print("bbbbbbbbbbbb", x_id, s.phase, s.state_name, "home:", s.home_nest, "loc:", s.location, "home_pop", nest_pop, happy_prob)
		probs = [1 - happy_prob, happy_prob]
	#print(x_id, s.phase, s.state_name, s.home_nest, s.candidate_nest, s.location, probs)
	c = random.choice(np.array(list(all_OTs[s.phase][s.state_name].keys())), p=probs)
	if s.state_name == "reverse_lead" and c == 'no_action':
		Ants[x_id].num_rev_tandems += 1
	return c

# Probabilistically picks a receiving ant based on the initiating ant
# return that receiving ant id
def Dy(action_type, x_id, src_nest):
	receiving_ants = []
	if action_type == "call":
		receiving_ants = [i for i in range(num_active) if (i in Nests[src_nest].ants_in_nest and i != x_id)]
	elif action_type == "carry":
		 # First try to transport non-active members
		receiving_ants = [i for i in range(num_active, num_ants) if (i in Nests[src_nest].ants_in_nest and i != x_id)]
		if len(receiving_ants) == 0:
			receiving_ants = [i for i in range(num_active) if (i in Nests[src_nest].ants_in_nest and i != x_id)]
	if len(receiving_ants) > 0:
		return random.choice(np.array(receiving_ants))
	else:
		return -1

def DNest(exclude_nests=[]):
	available = [i for i in range(num_nests) if i not in exclude_nests]
	return random.choice(np.array(available))

def adjust_phase(s, a):
	if s.home_nest == s.candidate_nest:
		return
	if s.phase == 'E' and s.state_name == "arrive" and a == "no_reject":
		s.phase = 'A'
	elif s.phase == 'A' and s.state_name == "at_nest" and a == "accept":
		s.phase = 'C'
	elif s.phase == 'C':
		if s.state_name == "quorum_sensing" and a == "quorum_met":
			s.phase = 'T'
			Nests[s.home_nest].committed_ants -= 1
			tmp = s.home_nest
			s.home_nest = s.candidate_nest
			s.candidate_nest = tmp
			#s.old_candidate_nest = s.candidate_nest
			Nests[s.home_nest].committed_ants += 1
		elif s.state_name == 'arrive' and a == 'no_reject':
			s.phase = 'A'
	elif s.phase == 'T' and s.state_name == "arrive" and a == "no_reject":
		s.phase = 'A'

def adjust_nests(x_id, s, a, new_nest=-1):
	global ind
	global tandem
	global transported
	#print(x_id,s.state_name,a,new_nest,s.candidate_nest)
	Nests[s.location].ants_in_nest.remove(x_id)
	if not is_brood(x_id):
		Nests[s.location].adult_ants_in_nest -= 1
	if s.state_name == "search":
		if a == "find":
			nnest = DNest([s.home_nest])
			s.old_candidate_nest = s.candidate_nest
			s.candidate_nest = nnest
			s.location = s.candidate_nest
	elif s.state_name == "arrive":
		s.location = s.candidate_nest
		if a == "reject" and s.phase != "T":
			s.candidate_nest = s.old_candidate_nest
			s.old_candidate_nest = -1
	# Note that the cases below should only be reached when a pair wise transition is successful
	# Also note that the last 3 cases below handle the receiving ant's nest and location changes, and 
	# the first 2 handle the initiating ant's. The last case below is only for a brood/passive's nest movements.
	elif a == "call":
		if s.state_name == "lead_forward":
			s.location = s.candidate_nest
		else:
			assert(new_nest > -1)
			s.old_candidate_nest = s.candidate_nest
			s.candidate_nest = new_nest
	elif a == "carry":
		if s.state_name == "transport":
			s.location = s.home_nest
		else: 
			assert(new_nest > -1)
			Nests[s.home_nest].committed_ants -= 1
			s.old_candidate_nest = -1
			s.candidate_nest = -1
			s.home_nest = new_nest
			s.phase = "E"
			s.location = new_nest
			Nests[s.location].committed_ants += 1
			if is_active(x_id) and (new_nest not in Ants[x_id].nests_visited):
				Ants[x_id].nests_visited.append(new_nest)
				transported += 1
	elif s.state_name == "follow":
		if a == "follow_find":
			#assert(s.candidate_nest > -1)
			#print(x_id, s.phase, s.state_name, s.home_nest, s.candidate_nest, s.old_candidate_nest, s.location, new_nest, a)
			assert(s.candidate_nest > -1)
			s.location = s.candidate_nest
		else: #get_lost while following, forget new candidate nest and go back to the old candidate nest
			s.candidate_nest = s.old_candidate_nest
			s.old_candidate_nest = -1
	Nests[s.location].ants_in_nest.append(x_id)
	if not is_brood(x_id):
		Nests[s.location].adult_ants_in_nest += 1


def main():
    if not os.path.exists(str(date.today())):
        os.makedirs(str(date.today()))
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    assert('ENVIRONMENT' in config)
    assert('ALGO' in config)
    assert('SETTINGS' in config)
    env = config['ENVIRONMENT']
    algo = config['ALGO']
    settings = config['SETTINGS']

    c_num_ants = [int(i) for i in env['num_ants'].split('|')]
    c_nest_qualities = env['nest_qualities'].split('|')
    c_lambda_sigmoid = [float(i) for i in env['lambda_sigmoid'].split('|')]

    c_pop_coeff = [float(i) for i in algo['pop_coeff'].split('|')]
    c_QUORUM_THRE = [float(i) for i in algo['QUORUM_THRE'].split('|')]
    c_QUORUM_OFFSET = [int(i) for i in algo['QUORUM_OFFSET'].split('|')]
    c_search_find = [float(i) for i in algo['search_find'].split('|')]
    c_follow_find = [float(i) for i in algo['follow_find'].split('|')]
    c_lead_forward = [float(i) for i in algo['lead_forward'].split('|')]
    c_transport = [float(i) for i in algo['transport'].split('|')]

    pl = int(settings['plot'])
    total_runs_per_setup = int(settings['total_runs_per_setup'])
    if pl:
    	assert(total_runs_per_setup == 1)
    c_num_rounds = int(settings['num_rounds'])
    c_percent_conv = float(settings['percent_conv'])
    c_persist_rounds = int(settings['persist_rounds'])

    global num_ants, num_active, num_passive, num_brood, nest_qualities, num_nests, lambda_sigmoid, pop_coeff, QUORUM_THRE, QUORUM_OFFSET, search_find, follow_find, lead_forward, transport, num_rounds, percent_conv, persist_rounds
    csvfilename = str(date.today())+"/results_"+sys.argv[1]+".csv"
    compact_csvname = str(date.today())+"/compact_results_"+sys.argv[1]+".csv"
    if os.path.exists(csvfilename):
        csvfile = open(csvfilename, "a")
        compact_csvfile = open(compact_csvname, "a")
    else:
        csvfile = open(csvfilename, "w")
        compact_csvfile = open(compact_csvname, "w")
    if not pl:
	    header = "nest_qualities, num_ants, pop_coeff, lambda_sigmoid, QUORUM_THRE, QUORUM_OFFSET, search_find, follow_find, lead_forward, transport, run_number, score, conv_nest, conv_nest_quality\n"
	    csvfile.write(header)
	    compact_header = "nest_qualities, pop_coeff, lambda_sigmoid, QUORUM_THRE, QUORUM_OFFSET, avg_score\n"
	    compact_csvfile.write(compact_header)
    num_rounds = c_num_rounds
    percent_conv = c_percent_conv
    persist_rounds = c_persist_rounds
    params = [c_num_ants, c_nest_qualities, c_lambda_sigmoid, c_pop_coeff, c_QUORUM_THRE, c_QUORUM_OFFSET, c_search_find, c_follow_find, c_lead_forward, c_transport]
    for param in list(itertools.product(*params)):
        (num_ants, _nest_qualities, lambda_sigmoid, pop_coeff, QUORUM_THRE, QUORUM_OFFSET, search_find, follow_find, lead_forward, transport) = param
        if (QUORUM_THRE==0.0 and QUORUM_OFFSET==0):
            pass
        nest_qualities = [float(i) for i in _nest_qualities.split(',')]
        #print(num_ants, nest_qualities, lambda_sigmoid, pop_coeff, search_find)
        num_nests = len(nest_qualities)
        assert(num_ants % 4 == 0)
        num_active = int(num_ants/4)
        num_passive = int(num_ants/4)
        num_broods = int(num_ants/2)
        setup_score = 0.0
        histn_all = [0] * 9
        disc_routes_all = [0] * 3
        for run_number in range(total_runs_per_setup):
        	sc, histn, disc_routes = execute(pl, run_number, csvfile)
        	setup_score += sc
        	histn_all = [sum(x) for x in zip(histn,histn_all)]
        	disc_routes_all = [sum(x) for x in zip(disc_routes, disc_routes_all)]
        histn_all = histn_all/sum(histn_all)
        disc_routes_all = disc_routes_all / np.sum(disc_routes_all)
        print(histn_all)
        print('Discovery routes:' , disc_routes_all)
        setup_score /= total_runs_per_setup
        if not pl:
        	compact_csvfile.write('"'+str(nest_qualities)+'",'+str(pop_coeff)+","+str(lambda_sigmoid)+","+str(QUORUM_THRE)+","+str(QUORUM_OFFSET)+","+str(setup_score)+"\n")
    csvfile.close()
    compact_csvfile.close()
if __name__ == "__main__":
	main()
