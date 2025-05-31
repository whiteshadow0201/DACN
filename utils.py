
# utils.py
import math
import random
import numpy as np
from queue import PriorityQueue
from copy import deepcopy
import torch
import torch.nn as nn
from collections import deque

def cantor_pairing(k1, k2):
    return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

def inverse_cantor(z):
    w = int((math.sqrt(8*z + 1) - 1) // 2)
    t = (w * (w + 1)) // 2
    k2 = z - t
    k1 = w - k2
    return k1, k2

def action_to_index(action, num_nodes):
    first = np.argmax(action[0])
    second = np.argmax(action[1])
    if first == second:
        raise ValueError("Action nodes must be distinct")
    return cantor_pairing(first, second)

def index_to_action(index, num_nodes):
    first, second = inverse_cantor(index)
    if first == second or first >= num_nodes or second >= num_nodes:
        raise ValueError("Invalid index or node count")
    action = np.zeros((2, num_nodes), dtype=np.float32)
    action[0, first] = 1
    action[1, second] = 1
    return action




def is_valid_index(index, num_nodes):
    first, second = inverse_cantor(index)
    return first != second and first < num_nodes and second < num_nodes

def sample_valid_index(action_space_size, num_nodes):
    while True:
        action_idx = random.randint(0, action_space_size - 1)
        if is_valid_index(action_idx, num_nodes):
            return action_idx


def sample_exploration_index(new_action_space_size, new_num_nodes, old_num_nodes):
    new_node_indices = []
    for i in range(new_num_nodes):
        for j in range(new_num_nodes):
            if i != j:
                if (i >= old_num_nodes or j >= old_num_nodes):
                    idx = cantor_pairing(i, j)
                    if idx >= new_action_space_size :
                        continue
                    new_node_indices.append(idx)

    if not new_node_indices:
        raise ValueError("No valid new-node-related actions found")
    return random.choice(new_node_indices)


class DQN(nn.Module):
    def __init__(self, state_size, action_space_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_space_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )

    def __len__(self):
        return len(self.buffer)


# 1. Environment Setup
# Environment class
class NetworkSecurityEnv:
    def __init__(self, G_new, attack_fn, goal="Data Server"):
        self.G_new = G_new
        self.attack_fn = attack_fn
        self.goal = goal
        self.nodes = [n for n in G_new.nodes if n not in ["Attacker", goal]]
        self.num_nodes = len(self.nodes)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
    def reset(self):
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        return self.state

    def step(self, action):
        honeypot_nodes = []
        G = deepcopy(self.G_new)

        for i in range(2):
            node_idx = np.argmax(action[i])
            if action[i, node_idx] == 0:
                node_idx = random.randint(0, self.num_nodes - 1)
            node = self.nodes[node_idx]
            honeypot = f"Honeypot {{{self.nodes[node_idx]}}}"
            honeypot_nodes.append(honeypot)
            G.add_node(honeypot)
            G.add_edge(node, honeypot, user=0.8, root=0.8)
            self.graph = deepcopy(G)
        path, captured = self.attack_fn(G, honeypot_nodes, self.goal)

        new_state = np.zeros(self.num_nodes, dtype=np.float32)
        reward = 0
        done = False

        if any(h in captured for h in honeypot_nodes):
            reward = 1
            done = True
        elif self.goal in captured:
            reward = -1
            done = True

        for node in captured:
            if node in self.node_to_idx:
                new_state[self.node_to_idx[node]] = 1

        self.state = new_state
        return new_state, reward, done, path, captured

    def get_action_space_size(self):
        return 2*self.num_nodes**2 + self.num_nodes


# Attacker's greedy attack with randomizer
def global_weighted_random_attack(graph, honeypot_nodes, goal):
    captured = {"Attacker"}
    path = ["Attacker"]

    while True:
        # Collect all uncaptured neighbors of compromised nodes
        neighbors = []
        edge_weights = []
        source_nodes = []

        for compromised_node in captured:
            for neighbor in graph.successors(compromised_node):
                if neighbor not in captured:
                    edge_data = graph[compromised_node][neighbor]
                    weight = edge_data['user'] + edge_data['root']
                    neighbors.append(neighbor)
                    edge_weights.append(weight)
                    source_nodes.append(compromised_node)

        if not neighbors:
            break

        # Normalize weights to probabilities
        total_weight = sum(edge_weights)
        if total_weight == 0:
            break
        probabilities = [w / total_weight for w in edge_weights]

        # Choose next node randomly based on probabilities
        chosen_idx = random.choices(range(len(neighbors)), weights=probabilities, k=1)[0]
        chosen_node = neighbors[chosen_idx]

        # Add to path and captured
        path.append(chosen_node)
        captured.add(chosen_node)

        # Check stopping conditions
        if chosen_node in honeypot_nodes or chosen_node == goal:
            break

    return path, captured

# Attacker's greedy attack with randomizer
def greedy_attack_priority_queue(graph, honeypot_nodes, goal):
    captured = {"Attacker"}
    path = ["Attacker"]
    pq = PriorityQueue()
    for neighbor in graph.successors("Attacker"):
        weight = max(graph["Attacker"][neighbor]['user'], graph["Attacker"][neighbor]['root'])
        randomizer = random.uniform(0, 1)  # Randomizer for tie-breaking
        pq.put((-weight, -randomizer, neighbor))  # Sort by -weight, -randomizer, neighbor

    while not pq.empty():
        neg_weight, neg_randomizer, to_node = pq.get()
        weight = -neg_weight
        randomizer = -neg_randomizer
        if to_node in honeypot_nodes:  # Stop at honeypot node
            path.append(to_node)
            captured.add(to_node)
            break
        if to_node not in captured:
            captured.add(to_node)
            path.append(to_node)
            if to_node == goal:
                break
            for next_node in graph.successors(to_node):
                if next_node not in captured:
                    next_weight = max(graph[to_node][next_node]['user'], graph[to_node][next_node]['root'])
                    next_randomizer = random.uniform(0, 1)  # New randomizer for each edge
                    pq.put((-next_weight, -next_randomizer, next_node))
    return path, captured




def evaluate_model(model, env, num_episodes=1000):
    successes = 0
    action_space_size = env.get_action_space_size()
    state_size = env.num_nodes
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        episode_honeypots = []  # Lưu vị trí honeypot trong episode

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor).squeeze(0)  # shape: [action_space_size]

                # Lọc q_values chỉ lấy index hợp lệ
                valid_indices = [idx for idx in range(action_space_size) if is_valid_index(idx, state_size)]
                valid_q_values = q_values[valid_indices]

                # Lấy chỉ số trong valid_indices có q_value max
                max_idx_in_valid = torch.argmax(valid_q_values).item()

                # Map về action_idx thực
                action_idx = valid_indices[max_idx_in_valid]

            action = index_to_action(action_idx, state_size)
            next_state, reward, done, path, captured = env.step(action)

            state = next_state
            honeypot_nodes = []
            for i in range(2):
                node_idx = np.argmax(action[i])
                honeypot_nodes.append(env.nodes[node_idx])
            print("Episode:", episode)
            if reward == 1:  # Honeypot bẫy được kẻ tấn công
                successes += 1
                print(path)
                print(f"Success\nHoneypots: {action}\nHoneypots connected to: {honeypot_nodes}\n")
                break
            elif reward == -1:  # Kẻ tấn công đạt mục tiêu
                print(path)
                print(f"Failed\nHoneypots: {action}\nHoneypots connected to: {honeypot_nodes}\n")
                break

    dsp = (successes / num_episodes) * 100
    print(f"\nDefense success probability: {dsp:.2f}%")



