import simpy 
import random 
import numpy as np
import gymnasium as gym  

class SimpleFactory:
    def __init__(self, env:simpy.Environment, resource_init:int=500):
        # Total resources
        self.resources = simpy.Container(env, capacity=resource_init, init=resource_init)
        # Resources (cache) scheduled for producing component c1
        self.resources_c1 = simpy.Container(env, capacity=3, init=0)
        # Resources (cache) scheduled for producing component c2
        self.resources_c2 = simpy.Container(env, capacity=4, init=0)
        # components c1
        self.c1 = simpy.Container(env, capacity=10, init=0)
        # components c2
        self.c2 = simpy.Container(env, capacity=10, init=0)
        # Products
        self.products = simpy.Container(env, capacity=500, init=0)


def resource_schedule(env:simpy.Environment, factory:SimpleFactory, action:int):
    check_time = 1
    if action==0: # send resource to produce c1
        if factory.resources.level >= 1:
            yield factory.resources.get(1) 
            dispatch_time = max(random.gauss(1, 0.1),0.7)
            yield env.timeout(dispatch_time)
            yield factory.resources_c1.put(1) 
        else:
            yield env.timeout(check_time)
        
    elif action==1: #send resource to produce c2
        if factory.resources.level >= 1:
            yield factory.resources.get(1)  
            dispatch_time = max(random.gauss(1,0.1), 0.7)
            yield env.timeout(dispatch_time)
            yield factory.resources_c2.put(1)
        else:
            yield env.timeout(check_time)
    
    elif action==2: # do nothing
        yield env.timeout(check_time)
    
    else:
        yield env.timeout(0)



def produce_c1(env:simpy.Environment, factory:SimpleFactory):
    check_time = 0.5 # check the container of resources for c1 every 0.5 second
    while True:
        if factory.resources_c1.level >= 2:
            yield factory.resources_c1.get(2) # use 2 units of resources 
            produce_time = max(random.gauss(3,0.3), 2.5)
            yield env.timeout(produce_time)
            yield factory.c1.put(1)
        else: yield env.timeout(check_time)
        


def produce_c2(env:simpy.Environment, factory:SimpleFactory):
    check_time = 0.5 # check the container of resources for c1 every 0.5 second
    while True:
        if factory.resources_c2.level >= 1:
            yield factory.resources_c2.get(1) # use 1 unit of resources 
            produce_time = max(random.gauss(1,0.1), 0.8)
            yield env.timeout(produce_time)
            yield factory.c2.put(1)
            current = env.now
        else: yield env.timeout(check_time)


def assemble(env:simpy.Environment, factory:SimpleFactory):
    check_time = 0.5 #check the containers of c1 and c2 every 0.5 second
    while True:
        if factory.c1.level >= 1 and factory.c2.level >=3:
            assemble_time = max(random.gauss(1.5,0.2),1)
            yield factory.c1.get(1)
            yield factory.c2.get(3)
            yield env.timeout(assemble_time)
            yield factory.products.put(1)
        else: yield env.timeout(check_time)

    



class SimpleFactoryGymEnv(gym.Env):
    def __init__(self, env_config=None): # The env_config parameter is for ray rllib environment
        super().__init__()
        self.simpy_env = simpy.Environment()
        self.factory = SimpleFactory(self.simpy_env, resource_init=500)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0,1,shape=(3,),dtype=np.float32)
        self.resource_init = 500
        self.max_episode_time = 800
        self.no_products_time = 0

    def _get_obs(self):
        return np.array([self.factory.resources.level/self.factory.resources.capacity, 
                         self.factory.resources_c1.level/self.factory.resources_c1.capacity, 
                         self.factory.resources_c2.level/self.factory.resources_c2.capacity], 
                        dtype=np.float32)
    
    def _get_info(self):
        return {"resources":self.factory.resources.level, 
                "resources_c1":self.factory.resources_c1.level,
                "resources_c2":self.factory.resources_c2.level, 
                "c1":self.factory.c1.level,
                "c2":self.factory.c2.level, 
                "products":self.factory.products.level,
                "current_time":self.simpy_env.now}
    
    def _check_products_status(self):
        """check if no more products can be produced"""
        if self.factory.resources.level==0: # No more resources
            if self.factory.c1.level < 1 or self.factory.c2.level < 3:
                self.no_products_time += 1

    def reset(self):
        self.simpy_env = simpy.Environment()
        self.factory = SimpleFactory(self.simpy_env, resource_init=self.resource_init)
        self.no_products_time = 0
        observation = self._get_obs()
        info = self._get_info()
        self.produce_c1_gen = produce_c1(self.simpy_env, self.factory)
        self.produce_c2_gen = produce_c2(self.simpy_env, self.factory)
        self.assemble_gen = assemble(self.simpy_env, self.factory)
        return observation, info

    
    def step(self, action):
        
        dispatch_cost = 0 if action==2 else 0.01 #dispatch cost 

        current_products = self.factory.products.level
        

        schedule_event = self.simpy_env.process(resource_schedule(self.simpy_env, self.factory, action))
        self.simpy_env.process(self.produce_c1_gen)
        self.simpy_env.process(self.produce_c2_gen)
        self.simpy_env.process(self.assemble_gen)

        self.simpy_env.run(until=schedule_event)
        
        observation = self._get_obs()
        info = self._get_info()

        reward = self.factory.products.level - current_products  
        

        self._check_products_status()

        terminated = (self.simpy_env.now >= self.max_episode_time) or (self.no_products_time>=5)    

        return observation, reward-dispatch_cost, terminated, False, info


