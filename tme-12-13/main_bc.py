from utils import *
from data_utils import *
from behavioral_cloning import *
from torch import *
import torch
import datetime
import yaml
import gym
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
# Load args

# behavioral cloning args
with open('./configs/behavioral_cloning.yaml', 'r') as stream:
    bc_args  = yaml.load(stream,Loader=yaml.Loader)

# load the environment (LunarLander)
with open('./configs/config_random_lunar.yaml', 'r') as stream:
    config  = yaml.load(stream,Loader=yaml.Loader)
env = gym.make(config['env'])
    
# file where are stored expert transitions
path = "./expert_data/"
file = "expert.pkl"

# expert data
expert_dataset = ExpertDataset(env, path+file, device)

# data loader
batch_size = bc_args['batch_size']
if batch_size<0:
    batch_size = len(expert_dataset)
expert_dataloader = DataLoader(expert_dataset, batch_size=batch_size)


# state dimension and nb_actions
input_size = env.observation_space.shape[0]
output_size = env.action_space.n


# network, criterion (log prob) and potimizer
net = Behavioral_Cloning(input_size, output_size, bc_args['layers']).to(device)
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=bc_args['lr'])


def train(optimizer, criterion, expert_dataloader, n_epochs = bc_args['epochs']):
    # training loop
    
    # Loop over epochs
    for epoch in range(n_epochs):
        # Training
        for states, actions in expert_dataloader:
            # Transfer to GPU
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            output = net(states.float())
            _, targets = actions.max(dim=1)
            loss = criterion(output, Variable(targets))
            loss.backward()
            optimizer.step()
    
# begin experimentations

nb_checks = 500
nb_episodes = 100
    
launch = True
    
if launch == True:

    # ---environment---#
    config = load_yaml("./configs/config_random_lunar.yaml")
    env = gym.make(config.env)
    if hasattr(env, "setPlan"):
        env.setPlan(config.map, config.rewards)
    tstart = str(time.time()).replace(".", "_")
    env.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    episode_count = config.nbEpisodes
    ob = env.reset()

    # ---agent---#
    agent_id = "BC"
    agent = net

    # ---yaml and tensorboard---#
    outdir = "./runs/"+ agent_id + "_" + tstart
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, "info.yaml"), config)
    writer = SummaryWriter(outdir)
    
    
    for chek in range(nb_checks):
        
        # train the agent for some epochs with the expert's data
        # Loop over epochs
        for epoch in range(bc_args["epochs"]):
            # Training
            for states, actions in expert_dataloader:
                # Transfer to GPU
                states, actions = states.to(device), actions.to(device)
                optimizer.zero_grad()
                output = net(states.float())
                _, targets = actions.max(dim=1)
                loss = criterion(output, Variable(targets))
                loss.backward()
                optimizer.step() 
        print("-------------------------")
        print("- loss :", loss.item())    
        agent = net
        
        # game's params
        rsum = 0
        mean = 0
        verbose = False
        itest = 0
        reward = 0
        done = False
        
        # play nb_episodes episodes
        for i in range(nb_episodes):
            done = False
            
            # play one episode
            while not (done):
                
                # get the action from the imitator
                probs = agent(torch.tensor(ob).to(device)) # generate distributions over states
                action = torch.argmax(nn.Softmax()(probs), dim=0).item() # action that maximize log likelihood

                ob_new, reward, done, _ = env.step(action)  # process action
                ob = ob_new
                # reward cumulé
                rsum += reward
                
                # if game is done
                if done:
                    agent.nbEvents = 0
                    mean += rsum
                    rsum = 0
                    ob = env.reset()
                    break
        print(f"{chek} mean_rewards={mean/nb_episodes}")
        writer.add_scalar("reward", mean/nb_episodes, chek)
        env.close()