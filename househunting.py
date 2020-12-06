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

bs = [0,1,6,11,16,21,26,31,36]

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
        self.discovery_route = -1
        self.first_transport = [0, -1] # [nest, round]

class State:
    def __init__(self, _state_name="at_nest", _home_nest=0, _candidate_nest=-1, _transitioned=False):
        self.state_name = _state_name # name of the state_name
        self.home_nest = _home_nest # default to 0
        self.candidate_nest = _candidate_nest 
        self.transitioned = _transitioned
        self.location = _home_nest
        self.phase = "E" # how much ant is committed to the candidate nest
        self.old_candidate_nest = -1 # only used for reject action
        self.terminate = 0

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

class Graph:
    def __init__(self, nest_array, edge_list):
        self.nests = nest_array
        self.edges = {}
        for edge in edge_list:
            edge_parts = edge.split(" ")
            node_1 = float(edge_parts[0])
            node_2 = float(edge_parts[1])
            length = float(edge_parts[2])
            if self.edges.get(node_1):
                self.edges[node_1].append((node_2, length))
            else:
                self.edges[node_1] = [(node_2, length)]
            if self.edges.get(node_2):
                self.edges[node_2].append((node_1, length))
            else:
                self.edges[node_2] = [(node_1, length)]

        for node in self.edges.keys():
            self.edges[node].sort()

    def get_probability_distribution(self, site_index):
        curr_edges = [1.0/i[1] for i in self.edges[site_index]]
        edge_sum = sum(curr_edges)
        prob_list = [(i*1.0/edge_sum) for i in curr_edges]
        return prob_list

    def get_average_nest_distance(self, site_index):
        curr_edges = [i[1] for i in self.edges[site_index]]
        edge_avg = sum(curr_edges)*1.0/len(curr_edges)
        return edge_avg

    def get_search_find_prob(self, ant_home_nest):
        home_avg = self.get_average_nest_distance(0)
        other_avg = self.get_average_nest_distance(ant_home_nest)
        if other_avg > home_avg:
            new_search_find = home_avg*home_avg*1.0/(other_avg*other_avg)*search_find
        else:
            new_search_find = search_find
        return [new_search_find, 1-new_search_find]

    def get_follow_find_prob(self, ant_home_nest):
        home_avg = self.get_average_nest_distance(0)
        other_avg = self.get_average_nest_distance(ant_home_nest)
        if other_avg > home_avg:
            new_follow_find = home_avg*home_avg*1.0/(other_avg*other_avg)*follow_find
        else:
            new_follow_find = follow_find
        return [new_follow_find, 1-new_follow_find]



Nests = {}
# Dictionary x_id : ant. Note: Ants[-1] is a place holder. 
# Ants[-1] has all members = NULL
Ants = {} 

global NestGraph
NestGraph = None

# A 2-level table corresponding to the input actions for each phase.
# First level is the current agent-state of receiving ant, and the second
# level is the action_type. The value of the second level dictionary item
# is a pair containing the resulting state, and a list of phases that 
# allow this input action for the receiving ant in the current agent-state

IT = { "at_nest":{"call":("follow", ["E","A","T"]),
       "carry":("at_nest", ["E","A","C","T"])}, 
       "search":{"carry":("at_nest", ["E","A","C","T"])}
     }

# Four 2-level table corresponding to the Output Actions for each
# phase.  First level is the current agent-state of initiating ant, and
# second is the action_type. The value of the second level
# dictionary item is the resulting agent-state.
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
OT_C = {"at_nest":{"search":"search", "recruit":"quorum_sensing"},  # phase entry point
        "search":{"find":"arrive", "no_action":"at_nest"}, 
        "lead_forward":{"call":"at_nest", "get_lost":"search", "terminate":"at_nest"},
        "arrive":{"reject":"search","no_reject":"at_nest"},
        "quorum_sensing":{"quorum_met":"transport","quorum_not_met":"lead_forward"}
       }
OT_T = {"at_nest":{"search":"search", "recruit":"transport"},
        "search":{"find":"arrive", "no_action":"at_nest"},
        "follow":{"follow_find":"arrive", "get_lost":"search"},
        "transport":{"carry":"reverse_lead", "stop_trans":"search", "terminate":"at_nest"}, # phase entry point
        "reverse_lead":{"no_action":"at_nest","delay":"reverse_lead"},
        "arrive":{"reject":"transport","no_reject":"at_nest"}
       }
all_OTs = {"E":OT_E, "A":OT_A, "C":OT_C, "T":OT_T}


# list of all the actions involving two ants
all_pair_actions = ["carry", "call"] 

def initiate_all_ants():
    # Put all active ants in search state first
    for i in range(num_active):
        # Ants[i] = Ant(_x_id=i, _cur_state = State(_state_name="search", _home_nest=0, _candidate_nest=-1, _transitioned=False))
        Ants[i] = Ant(_x_id=i)
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
    # Generate the graph
    global NestGraph
    NestGraph = Graph(Nests,c_edges)
    # should have at least 1 home nest and 1 candidate nest
    assert(len(Nests) >= 2)
    # Initiate all ants to be in home nest
    #Nests[0].committed_ants = num_ants
    #Nests[0].ants_in_nest = [x for x in range(0,num_ants)]
    #Nests[0].adult_ants_in_nest = num_active+num_passive


def print_all_ants_states(startid, endid):
    for x_id in range(startid, endid):
            ant = Ants[x_id]
            print(x_id, ant.cur_state.phase, ant.cur_state.state_name, "home:", ant.cur_state.home_nest, "cand:", ant.cur_state.candidate_nest, "old_cand:", ant.cur_state.old_candidate_nest, "loc:", ant.cur_state.location, ant.proposed_action.action_type, "recv:", ant.proposed_action.receiving, ant.nests_visited, ant.num_transports)

def print_all_nests_info(y, y_ants_in_nest):
    nests_visited = [0]*num_nests
    num_transports_by_nest_visits = [0]*num_nests
    for x_id in range(num_active):
        nests_visited[len(Ants[x_id].nests_visited)-1] += 1
        num_transports_by_nest_visits[len(Ants[x_id].nests_visited)-1] += Ants[x_id].num_transports
    for nest_id in range(num_nests):
        nest = Nests[nest_id]
        y[nest_id].append(nest.committed_ants)
        y_ants_in_nest[nest_id].append(len(nest.ants_in_nest))
        #print("Nest quality", nest.quality, ", ants committed:", nest.committed_ants, ", ants in nest:", len(nest.ants_in_nest), ", adults in nest:", nest.adult_ants_in_nest)
    # print("# ants seeing 1,2,3... nests", nests_visited[1:], "num_transports by # of nests visited:", num_transports_by_nest_visits)
    return nests_visited, num_transports_by_nest_visits

def is_active(x_id):
    return x_id < num_active

def is_passive(x_id):
    return (x_id >= num_active and x_id < num_active + num_passive)

def is_brood(x_id):
    return (x_id >= num_active + num_passive and x_id < num_ants)

def sigmoid(x):
    return 1/(1+np.exp(-x*lambda_sigmoid))

# This function executes on round of transitions. One action proposed by each ant
def execute_one_round(r):
    for x_id in range(num_ants):
        ant = Ants[x_id]
        ant.cur_state.transitioned = False
        if x_id < num_active:
            if ant.cur_state.state_name == 'quorum_sensing' and ant.first_transport[1] == -1:
                ant.first_transport = [ant.cur_state.candidate_nest, r]
                # print(x_id, ant.cur_state.home_nest)
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
    elif a.action_type == "find":
        nnest = DNest([s.home_nest])
        s.old_candidate_nest = s.candidate_nest
        s.candidate_nest = nnest
    elif a.action_type == "terminate":
        s.terminate = 0
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
                # print(x.x_id, y.x_id, state_y.state_name, state_y.location, state_x.candidate_nest, action_type)
                success, new_y_state_name = input_transition(y.x_id, state_y, action_type)
                if success: 
                    output_transition(x.x_id, state_x, action_type)
                    nnest = state_x.candidate_nest
                    if action_type == "carry":
                        nnest = state_x.home_nest
                        x.num_transports += 1
                    ret = adjust_nests(y.x_id, state_y, action_type, new_nest = nnest)
                    if ret == "":
                        state_y.state_name = new_y_state_name
                    state_y.transitioned = True
                    return True, state_x, state_y   
            elif y.x_id == -1 or (state_x.state_name == "transport" and state_y.state_name == "transport") or (state_x.state_name == "lead_forward" and state_y.state_name == "lead_forward"):
                    state_x.terminate += 1
                    # state_x.state_name = "at_nest"
                    # state_x.phase = 'E'
                    # adjust_nests(x.x_id, state_x, action_type, state_x.home_nest)
                    # state_x.transitioned = True
                    # return False, state_x, state_y
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
        probs = NestGraph.get_search_find_prob(s.home_nest)
    elif s.state_name == "follow":
        probs = NestGraph.get_follow_find_prob(s.home_nest)
    elif s.state_name == "lead_forward":
        probs = [lead_forward, 1-lead_forward, 0]
        if s.terminate == 10:
            probs = [0,0,1]
    elif s.state_name == "transport":
        probs = [transport, 1-transport, 0]
        if s.terminate == 10:
            probs = [0,0,1]
    elif s.state_name == "reverse_lead":
        probs = [1-transport,transport]
    elif s.state_name == "quorum_sensing":
        assert(s.phase == "C")
        probs = [0, 1]
        if s.candidate_nest != s.home_nest and Nests[s.candidate_nest].adult_ants_in_nest > (QUORUM_THRE*(num_active+num_passive)+QUORUM_OFFSET):
            probs = [1,0]
    elif s.state_name == "arrive":
        # Some stats
        if s.candidate_nest not in Ants[x_id].nests_visited:
            Ants[x_id].nests_visited.append(s.candidate_nest)
            if Ants[x_id].proposed_action.action_type == 'follow_find':
                tandem += 1
                Ants[x_id].discovery_route = 1
            elif Ants[x_id].proposed_action.action_type == 'find':
                ind += 1
                Ants[x_id].discovery_route = 0
        # choose probabilities
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
        # if s.phase == "A" and s.candidate_nest == s.home_nest:
        #     happy_prob = 0
        # else:
        nest_q = Nests[s.location].quality / 4
        nest_pop = len(Nests[s.location].ants_in_nest) / num_ants
        # nest_pop = Nests[s.location].adult_ants_in_nest / num_ants
        happy_prob = sigmoid(nest_q + pop_coeff*nest_pop)
            #print("bbbbbbbbbbbb", x_id, s.phase, s.state_name, "home:", s.home_nest, "loc:", s.location, "home_pop", nest_pop, happy_prob)
        probs = [1 - happy_prob, happy_prob]
    # print(x_id, s.phase, s.state_name, s.home_nest, s.candidate_nest, s.location, probs)
    c = random.choice(np.array(list(all_OTs[s.phase][s.state_name].keys())), p=probs)
    if s.state_name == "reverse_lead" and c == 'no_action' and random.random() < 0.05:
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
    # The simple case with the existing four nests in the config file, which currrently 
    # have values 0.5, 1, 1.5, 2 

    #return np.asscalar(np.random.choice(np.array(available), 1, p=NestGraph.get_probability_distribution(exclude_nests[0])))
    return random.choice(np.array(available))

def adjust_phase(s, a):
    # if s.home_nest == s.candidate_nest:
    #     return
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
            Nests[s.home_nest].committed_ants += 1
        elif s.state_name == 'arrive' and a == 'no_reject':
            s.phase = 'A'
    elif s.phase == 'T' and s.state_name == "arrive" and a == "no_reject":
        s.phase = 'A'

def adjust_nests(x_id, s, a, new_nest=-1):
    global ind
    global tandem
    global transported
    Nests[s.location].ants_in_nest.remove(x_id)
    ret = ""
    if not is_brood(x_id):
        Nests[s.location].adult_ants_in_nest -= 1
    if a not in all_pair_actions:
        if s.state_name == "search":
            if a == "find":
                # nnest = DNest([s.home_nest])
                # s.old_candidate_nest = s.candidate_nest
                # s.candidate_nest = nnest
                s.location = s.candidate_nest
        elif s.state_name == "arrive":
            s.location = s.candidate_nest
            if a == "reject" and s.phase != "T":
                s.candidate_nest = s.old_candidate_nest
                s.old_candidate_nest = -1
        elif s.state_name == "follow":
            if a == "follow_find":
                #print(x_id, s.phase, s.state_name, s.home_nest, s.candidate_nest, s.old_candidate_nest, s.location, new_nest, a)
                assert(s.candidate_nest > -1)
                s.location = s.candidate_nest
            else: #get_lost while following, forget new candidate nest and go back to the old candidate nest
                s.candidate_nest = s.old_candidate_nest
                s.old_candidate_nest = -1
        elif a == "terminate":
            s.location = s.home_nest
    # Note that the cases below should only be reached when a pair wise transition is successful
    # Also note that the last 3 cases below handle the receiving ant's nest and location changes, and 
    # the first 2 handle the initiating ant's. The last case below is only for a brood/passive's nest movements.
    else:
        if a == "call":
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
                if not is_active(x_id) or s.home_nest == new_nest:
                    Nests[s.home_nest].committed_ants -= 1
                    s.old_candidate_nest = -1
                    s.candidate_nest = -1
                    s.home_nest = new_nest
                    s.phase = "E"
                    s.location = new_nest
                    Nests[s.location].committed_ants += 1
                else:
                    s.old_candidate_nest = -1
                    s.candidate_nest = new_nest
                    s.phase = "E"
                    ret = "at_nest"
                    s.location = new_nest
                if is_active(x_id) and (new_nest not in Ants[x_id].nests_visited):
                    Ants[x_id].nests_visited.append(new_nest)
                    transported += 1
                    Ants[x_id].discovery_route = 2
        
    Nests[s.location].ants_in_nest.append(x_id)
    if not is_brood(x_id):
        Nests[s.location].adult_ants_in_nest += 1
    return ret

# main simulation function
def execute(plot, run_number, csvfile, is_validation = True):
    global ind
    global tandem
    global transported
    x = np.arange(0, num_rounds)
    y = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    y_ants_in_nest = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]   
    
    initiate_nests()
    initiate_all_ants()
    # initiate_all_ants_random_loc()

    best_nest = max(nest_qualities)
    best_nest_id = nest_qualities.index(best_nest)
    conv_start = [-1]*num_nests
    score = 0
    conv_nest = -1
    conv_nest_quality = -100

    ind = 0
    tandem = 0
    transported = 0
    broods_in_better_nest = 0
    ants_in_better_nest = 0
    for round in range(num_rounds):
        #print(round)
        # print([(len(Nests[0].ants_in_nest) - Nests[0].adult_ants_in_nest), (len(Nests[1].ants_in_nest) - Nests[1].adult_ants_in_nest), (len(Nests[2].ants_in_nest) - Nests[2].adult_ants_in_nest)])
        
        execute_one_round(round) 
        # if round % 500 == 0:
        #     print_all_ants_states(0, num_ants)
        nest_visits, transports_by_visits = print_all_nests_info(y, y_ants_in_nest)
        if len(Nests[0].ants_in_nest) == 0 and not plot:
            if num_nests >= 3:
                broods_in_better_nest = 1.*(len(Nests[best_nest_id].ants_in_nest) - Nests[best_nest_id].adult_ants_in_nest) / num_broods
                ants_in_better_nest = 1.*len(Nests[best_nest_id].ants_in_nest) / num_ants
            if is_validation:
                break
        for nest_id in range(1,num_nests):
            if y_ants_in_nest[nest_id][-1] > percent_conv * num_ants: #if converged
                if conv_start[nest_id] == -1:
                    conv_start[nest_id] = round
                else:
                    if round - conv_start[nest_id] == persist_rounds:
                        score = 1/(conv_start[nest_id]+1)
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
        row += ", "+str(search_find)+", "+str(follow_find)+", "+str(lead_forward)+", "+str(transport)+", "+str(run_number)\
        +", "+str(score)+", "+str(conv_nest)+", "+str(conv_nest_quality)+", "
        row += str(round) + ", " + str(broods_in_better_nest)+","+str(len(Nests[2].ants_in_nest))+"\n"
        csvfile.write(row)
        # print("# ants seeing 1,2,3... nests", nests_visited[1:], "num_transports by # of nests visited:", num_transports_by_nest_visits)
    
        recruit_acts = [(Ants[i].num_tandems + Ants[i].num_transports + Ants[i].num_rev_tandems) for i in range(num_active)]
        n,_,_ = plt.hist(recruit_acts, bins=bs)
        type_recruitments = [0] * 7
        for i in range(num_active):
            a = Ants[i].num_tandems
            b = Ants[i].num_transports
            c = Ants[i].num_rev_tandems
            if a and not b:
                type_recruitments[0] += 1
            elif b and not a and not c:
                type_recruitments[1] += 1
            elif a and b and not c:
                type_recruitments[2] += 1
            elif b and c and not a:
                type_recruitments[3] += 1
            elif c and not b:
                type_recruitments[4] += 1
            elif a and b and c:
                type_recruitments[5] += 1
            elif not a and not b and not c:
                type_recruitments[6] += 1

        recruit_per_discovery_route = [0] * 3
        for i in range(num_active):
            for route in range(3):
                if Ants[i].discovery_route == route and (Ants[i].num_tandems+Ants[i].num_transports) > 0:
                    recruit_per_discovery_route[route] += 1

        # all_disc = ind + tandem + transported
        recruit_per_discovery_route = [recruit_per_discovery_route[i]for i in range(3)]

        accuracy = 0
        if conv_nest == -1 :#or len(Ants[0].nests_visited) <= 2:
            accuracy = -1
            conv_nest = 0
        elif conv_nest_quality  == best_nest:
            accuracy = 1
        ac2 = -1
        if conv_nest_quality > 0:
            # ac2 = (conv_nest_quality == best_nest)
            ac2 = 1
        return score, len(Nests[1].ants_in_nest)*1./num_ants, \
        (broods_in_better_nest, round, ants_in_better_nest), n, \
        [ind, tandem, transported], type_recruitments, recruit_per_discovery_route, \
        [conv_nest_quality > 0, conv_nest_quality == best_nest, \
         [len(Ants[i].nests_visited) for i in range(num_active)], \
         nest_visits, transports_by_visits], conv_nest
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

def main():
    if not os.path.exists(str(date.today())):
        os.makedirs(str(date.today()))
    config = configparser.ConfigParser()
    config.read('ants.ini')
    assert('ENVIRONMENT' in config)
    assert('ALGO' in config)
    assert('SETTINGS' in config)
    env = config['ENVIRONMENT']
    algo = config['ALGO']
    settings = config['SETTINGS']

    c_num_ants = [int(i) for i in env['num_ants'].split('|')]
    c_nest_qualities = env['nest_qualities'].split('|')

    global c_edges
    c_edges = env['graph_edges'].split("|")

    c_lambda_sigmoid = [float(i) for i in algo['lambda_sigmoid'].split('|')]
    c_pop_coeff = [float(i) for i in algo['pop_coeff'].split('|')]
    c_QUORUM_THRE = [float(i) for i in algo['QUORUM_THRE'].split('|')]
    c_QUORUM_OFFSET = [int(i) for i in algo['QUORUM_OFFSET'].split('|')]
    c_search_find = [float(i) for i in algo['search_find'].split('|')]
    c_follow_find = [float(i) for i in algo['follow_find'].split('|')]
    c_lead_forward = [float(i) for i in algo['lead_forward'].split('|')]
    c_transport = [float(i) for i in algo['transport'].split('|')]

    pl = int(settings['plot'])
    total_runs_per_setup = int(settings['total_runs_per_setup'])
    # if pl:
    #     assert(total_runs_per_setup == 1)
    c_num_rounds = int(settings['num_rounds'])
    c_percent_conv = float(settings['percent_conv'])
    c_persist_rounds = int(settings['persist_rounds'])

    global num_ants, num_active, num_passive, num_broods, nest_qualities, num_nests, lambda_sigmoid, pop_coeff, QUORUM_THRE, QUORUM_OFFSET, search_find, follow_find, lead_forward, transport, num_rounds, percent_conv, persist_rounds
    csvfilename = str(date.today())+"/results_"+sys.argv[1]+".csv"
    compact_csvname = str(date.today())+"/compact_results_"+sys.argv[1]+".csv"
    splits_csvname = str(date.today())+"/splits_results_"+sys.argv[1]+".csv"
    if os.path.exists(csvfilename):
        csvfile = open(csvfilename, "a")
        compact_csvfile = open(compact_csvname, "a")
        splits_csvfile = open(splits_csvname, "a")
    else:
        csvfile = open(csvfilename, "w")
        compact_csvfile = open(compact_csvname, "w")
        splits_csvfile = open(splits_csvname, "w")
    if not pl:
        header = "nest_qualities, num_ants, pop_coeff, lambda_sigmoid, QUORUM_THRE, QUORUM_OFFSET, search_find, follow_find, lead_forward, transport, run_number, score, conv_nest, conv_nest_quality, avg_rounds_til_empty, avg_brood_good, avg_pop_good_nest\n"
        csvfile.write(header)
        compact_header = "nest_qualities, pop_coeff, lambda_sigmoid, QUORUM_THRE, QUORUM_OFFSET, avg_score, best_count, conv_count, search_find, avg_rounds_til_empty, avg_ants, avg_broods, avg_transports, conv_dist, conv_score\n"
        compact_csvfile.write(compact_header)
        splits_header = "num_ants, num_active, num_passive, num_broods, pop_coeff, \% Observed, \% Predicted, SD, P\n"
        splits_csvfile.write(splits_header)
    num_rounds = c_num_rounds
    percent_conv = c_percent_conv
    persist_rounds = c_persist_rounds

    histn_all = []
    disc_routes_all = []
    type_recruitments_all = []
    recruit_per_discovery_route_all = []
    visits_all = []
    
    visits_colony_all = []
    nests_visited_colony_all = []
    in_left_nest_all = []
    params = [c_num_ants, c_nest_qualities, c_lambda_sigmoid, c_pop_coeff, c_QUORUM_THRE, c_QUORUM_OFFSET, c_search_find, c_follow_find, c_lead_forward, c_transport]
    
    for param in list(itertools.product(*params)):
        num_transports_by_nest_visits = []
        splits = []
        rounds_til_empty_all = []
        ants_in_better_nest_all = []
        ac_colony_all = []
        best_all = []
        accuracy_all = []
        (num_ants, _nest_qualities, lambda_sigmoid, pop_coeff, QUORUM_THRE, QUORUM_OFFSET, search_find, follow_find, lead_forward, transport) = param
        if (QUORUM_THRE==0.0 and QUORUM_OFFSET==0):
            pass
        nest_qualities = [float(i) for i in _nest_qualities.split(',')]
        # print(num_ants, nest_qualities, lambda_sigmoid, pop_coeff, search_find)
        num_nests = len(nest_qualities)
        observed = 0
        print(num_ants)
        if num_ants >20 :
            assert(num_ants % 4 == 0)
            num_active = int(num_ants/4)
            num_passive = int(num_ants/4)
            num_broods = int(num_ants/2)
        elif num_ants == 1:    
            num_ants = 326
            num_active = 70
            num_passive = 28
            num_broods = 228
            # pop_coeff = 0.45
            observed = 0.61
        elif num_ants == 2:    
            num_ants = 244
            num_active = 59
            num_passive = 74
            num_broods = 111
            # pop_coeff = 0.35
            observed = 0.80
        elif num_ants == 3:
            num_ants = 263
            num_active = 62
            num_passive = 95
            num_broods = 106
            # pop_coeff = 0.4
            observed = 0.99
        elif num_ants == 4:
            num_ants = 301
            num_active = 67
            num_passive = 42
            num_broods = 192
            # pop_coeff = 0.4
            observed = 0.98
        elif num_ants == 5:
            num_ants = 202
            num_active = 53
            num_passive = 88
            num_broods = 61
            # pop_coeff = 0.15
            observed = 1.0
        elif num_ants == 6:
            num_ants = 347
            num_active = 73
            num_passive = 101
            num_broods = 173
            # pop_coeff = 0.3
            observed = 0.02


        setup_score = 0.0

        conv_cnt_by_nest_all = [0]*num_nests
        for run_number in range(total_runs_per_setup):
            sc, in_left_nest, (broods_in_better_nest, rounds_til_empty, ants_in_better_nest), \
            histn, disc_routes, type_recruitments, recruit_per_discovery_route, ac2, conv_nest_id = \
            execute(pl, run_number, csvfile)

            setup_score += sc
            # in_left_nest_all.append(in_left_nest)
            splits.append(broods_in_better_nest)
            rounds_til_empty_all.append(rounds_til_empty)
            ants_in_better_nest_all.append(ants_in_better_nest)
            conv_cnt_by_nest_all[conv_nest_id] += 1

            # Data collect
            histn_all.append([1.*histn[i]/num_active for i in range(len(histn))])
            disc_routes_all.append([1.*disc_routes[i]/num_active for i in range(len(disc_routes))])
            accuracy_all.append(broods_in_better_nest)
            # @visits_all For now we're not actually using this statistic, not sure what it was supposed to mean in Jiajia's code
            visits_all.append(1)

            # if 1:
                # accuracy_all.append(1)
                # visits_all.append(1)

            if ac2[0] > -1:
                ac_colony_all.append(ac2[0])
                best_all.append(ac2[1])
                visits_colony_all.append(ac2[2])
                nests_visited_colony_all.append(ac2[3])
                num_transports_by_nest_visits.append(ac2[4])
                
            # print('type_recruitments', type_recruitments)
            # if sum(type_recruitments[:-1]) > 0:
            #     type_recruitments_all.append([1.*type_recruitments[i]/sum(type_recruitments[:-1]) for i in range(6)])
            # recruit_per_discovery_route_all.append([1.*recruit_per_discovery_route[i]/num_active for i in range(len(recruit_per_discovery_route))])
           
        all_transports = [sum(x) for x in num_transports_by_nest_visits]
        print(str(lambda_sigmoid), str(search_find), str(nest_qualities), np.mean(all_transports), np.percentile(all_transports, 25), np.percentile(all_transports, 75))
        setup_score /= total_runs_per_setup
        # print(in_left_nest_all)
        conv_cnt_by_nest_all = np.array(conv_cnt_by_nest_all)*1.0/total_runs_per_setup
        conv_accuracy_score = np.dot(np.array(nest_qualities)/(max(nest_qualities)-min(nest_qualities)), conv_cnt_by_nest_all)
        
        if not pl:
            compact_csvfile.write('"'+str(nest_qualities)+'",'+str(pop_coeff)+\
                                  ","+str(lambda_sigmoid)+","+str(QUORUM_THRE)+\
                                  ","+str(QUORUM_OFFSET)+","+str(setup_score)+\
                                  ","+str(sum(best_all))+","+str(sum(ac_colony_all))+\
                                  ","+str(search_find)+","+str(np.mean(rounds_til_empty_all))+\
                                  ","+str(np.mean(ants_in_better_nest_all))+","+str(np.mean(splits))+\
                                  ","+str(np.mean(all_transports))+","+\
                                  str(conv_cnt_by_nest_all)+","+str(conv_accuracy_score)+"\n") 
            #+","+str(len(accuracy_all)*1./total_runs_per_setup)+"\n")
        # print(len(ac_colony_all))

        ### Below is for splits among 1 good and 1 poor nest
        if np.mean(splits) > observed:
            pvalue = 1.*len([splits[i] for i in range(total_runs_per_setup) if splits[i] <= observed ]) / total_runs_per_setup
        else:
            pvalue = 1.*len([splits[i] for i in range(total_runs_per_setup) if splits[i] >= observed ]) / total_runs_per_setup
        print(np.mean(splits), np.std(splits), pvalue)
        splits_header = str(num_ants)+','+ str(num_active)+','+ str(num_passive)+','+ str(num_broods)+','+ str(pop_coeff)+','+ str(observed)+',' + str(np.mean(splits))+','+str(np.std(splits))+ ','+str(pvalue)+'\n'
        splits_csvfile.write(splits_header)
        print(np.mean(accuracy_all), len(accuracy_all), np.percentile(visits_all,25), np.percentile(visits_all,50), np.percentile(visits_all,75))
        
    ### Build the percentage ants of different types of recruitment
    # fig, ax = plt.subplots()
    # counts = [sum(x) for x in zip(*num_transports_by_nest_visits)]
    # nests_counts = [sum(x) for x in zip(*nests_visited_colony_all)]
    # counts = counts[1:]/np.sum(counts)
    # nests_counts = nests_counts[1:]/np.sum(nests_counts)
    # np.pad(nests_counts, (0, len(counts)-len(nests_counts)), 'constant')
    # x_labels = [i for i in range(1, len(counts)+1)]
    # x_pos = np.arange(len(x_labels))
    # print("types of recruitment", x_pos, counts)
    # ax.bar(x_pos-0.1, counts, width=0.2, color='r', align='center', label='Number of Transports')
    # ax.bar(x_pos+0.1, nests_counts, width=0.2, color='b', align='center', label='Individual Ants')
    # ax.set_ylabel('Percentage')
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(x_labels)
    # ax.set_xlabel('Number of Nests Visited')
    # ax.yaxis.grid(False)
     # Save the figure and show
    # plt.tight_layout()
    # plt.savefig(str(len(counts))+'nests')
    # plt.show()
    # plt.close()
    # print('nest visits', np.mean(np.array(visits_colony_all)), len(visits_colony_all), np.percentile(visits_colony_all,25), np.percentile(visits_colony_all,50), np.percentile(visits_colony_all,75))
    
    # For recruitment acts
    error = []
    counts = []
    for i in range(len(bs)-1):
        count = [histn_all[r][i] for r in range(len(histn_all))]
        counts.append(np.mean(count))
        error.append(np.std(count))
    print('Discovery routes:' , recruit_per_discovery_route_all)

    fig, ax = plt.subplots()
    x_labels = ['0','1-5','6-10','11-15','16-20','21-25','26-30','31-35']
    x_pos = np.arange(len(x_labels))
    ax.bar(x_pos, counts, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xlabel('Number of Recruitment Acts')
    ax.set_ylabel('Percentage of Workers')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_recruit_act')
    # plt.show()
    plt.close()

    # # Build the percentage ants of different types of recruitment
    # fig, ax = plt.subplots()
    # counts = []
    # error = []
    # for i in range(6):
    #     count = [type_recruitments_all[r][i] for r in range(len(type_recruitments_all))]
    #     counts.append(np.mean(count))
    #     error.append(np.std(count))
    # x_labels = ['FTRs\nand\nTRTs','Transports\nonly','FTRs\nand\ntransports','RTRs\nand\ntransports','RTRs\nonly','FTRs\nand\ntransports\nand\nRTRs']
    # x_pos = np.arange(len(x_labels))
    # ax.bar(x_pos, counts, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax.set_ylabel('Percentage of Workers')
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(x_labels)
    # ax.yaxis.grid(True)
    #  # Save the figure and show
    # plt.tight_layout()
    # plt.savefig('type_recruitments')
    # # plt.show()
    # plt.close()

    # # Build the discovery route figure
    # fig, ax = plt.subplots()
    # y1 = []
    # y2 = []
    # for i in range(3):
    #     y1.append(np.mean([disc_routes_all[r][i] for r in range(len(disc_routes_all))]))
    #     y2.append(np.mean([recruit_per_discovery_route_all[r][i] for r in range(len(recruit_per_discovery_route_all))]))

    # x_labels = ['Independent', 'Followers', 'Transportees']
    # x_pos = np.arange(len(x_labels))
    # ax.bar(x_pos-0.1, y1, width=0.2, color='r', align='center')
    # ax.bar(x_pos+0.1, y2, width=0.2, color='b', align='center')
    # ax.set_ylabel('Percentage of Active Workers')

    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(x_labels)
    # ax.yaxis.grid(False)

    # # Save the figure and show
    # plt.tight_layout()
    # plt.savefig('discovery_recruit')
    # for i, v in enumerate(y1):
    #     plt.text(x_pos[i]-0.2, v + 0.01, "{00:.00f}%".format(round(v*100,4)))
    # for i, v in enumerate(y2):
    #     plt.text(x_pos[i], v + 0.01, "{00:.00f}%".format(round(v*100,4)))
    # plt.close()
    # # plt.show()

    csvfile.close()
    compact_csvfile.close()
    splits_csvfile.close()
if __name__ == "__main__":
    main()
