import neat
import pickle
import random
import gym
import numpy as np
import cv2

env = gym.make('VideoPinball-v0')

def noise(img,f):
    row,col,ch = img.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)
    noise = img + img*gauss*f
    return noise

img_arr = []

def eval_genomes(genomes,config):
    for genome_id,genome in genomes:
        obs = env.reset()
        x,y,c = env.observation_space.shape

        x = int(x/8)
        y = int(y/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome,config)

        current_fitness = 0
        frame = 0
        counter = 0
        done = False
        while not done:
            frame += 1
            factor = 0.5
            
            obs = np.uint8(noise(obs,factor))
            obs = cv2.resize(obs,(x,y))
            obs = cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)

            img_arr = np.ndarray.flatten(obs)
            net_output = net.activate(img_arr)

            numerical_input = net_output.index(max(net_output))
            obs,rew,done,info = env.step(numerical_input)
            current_fitness += rew

            if rew>0:
                counter = 0
            else:
                counter +=1
            
            env.render()
            if done or counter==250:
                done = True
                print(genome_id,'   Fitness: ',current_fitness)
            genome.fitness = current_fitness

def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward.txt')
    pop = neat.Population(config)
    pop.add_reporter(neat.Checkpointer(10))

    winner = pop.run(eval_genomes)
    with open('winner_pinball_1.pkl','wb') as output:
        pickle.dump(winner,output,1)

main()