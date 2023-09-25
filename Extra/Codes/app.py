import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import torch
torch.manual_seed(0)
import gym

class CartPole(torch.nn.Module):
  def __init__(self, state_size=4, action_size=2, hidden_size=32):
    super(CartPole, self).__init__()
    self.scores = []
    self.environment = gym.make('CartPole-v0')
    self.environment.seed(20)
    if torch.cuda.is_available():
      self.device = torch.device("cuda:0")
    else:
      self.device = torch.device("cpu")

  def constructModel(self):
    self.hidden1 = torch.nn.Linear(4, 32)
    self.hidden2 = torch.nn.Linear(32, 2)
    
  def forward(self, state):
      x = torch.nn.functional.relu(self.hidden1(state))
      x = self.hidden2(x)
      return torch.nn.functional.softmax(self.hidden2(torch.nn.functional.relu(self.hidden1(state))), dim=1)
  
  def act(self, state):
      state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
      probs = self.forward(state).cpu()
      model = torch.distributions.Categorical(probs)
      action = model.sample()
      return action.item(), model.log_prob(action)

  def train(self):
    self.scores_deque = deque(maxlen=100)
    optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
    self.train_(episodes=5000)

  def plotScores(self):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(self.scores)+1), self.scores)
    plt.ylabel('Score')
    plt.xlabel('Iteration')
    plt.title('Cart Pole Using Policy Gradient Method')
    plt.show()

  def setPolicy(self):
      self.policy =  model.to(model.device)

  def train_(self,episodes=1000):
      gamma=1.0;
      for k in range(1, episodes):
          loss = []
          saved_log_probs = []
          rewards = []
          state = self.environment.reset()
          for t in range(1000):
              action, log_prob = self.act(state)
              saved_log_probs.append(log_prob)
              state, reward, done, _ = self.environment.step(action)
              rewards.append(reward)
              if done:
                  break
          self.scores_deque.append(sum(rewards))
          self.scores.append(sum(rewards))
          discounts = [gamma ** i for i in range(len(rewards) + 1)]
          R = sum([a * b for a,b in zip(discounts, rewards)])
          
          for log_prob in saved_log_probs:
              loss.append(-log_prob * R)
          loss = torch.cat(loss).sum()
          
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          
          if (k%50 == 0):
            print('Avg Score of episode ', k, 'is', np.mean(self.scores_deque))
          if np.mean(self.scores_deque) >= 200:
            break

if __name__ == "__main__":
  model = CartPole()
  model.constructModel()
  model.setPolicy()
  model.train()
  model.plotScores()