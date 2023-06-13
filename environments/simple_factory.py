import simpy 
import random 

class SimpleFactory:
    """
    A factory producing components c1 and c2 using resources 
    and then assembling the components into products
    """
    def __init__(self, env:simpy.Environment, resource_init:int=500):
        self.resources = simpy.Container(env, capacity=resource_init, init=resource_init)
        # Total resources
        self.c1 = simpy.Container(env, capacity=resource_init, init=0)
        # Components 1
        self.c2 = simpy.Container(env, capacity=resource_init, init=0)
        # Components 2
        self.products = simpy.Container(env, capacity=resource_init, init=0)




def produce_component(env:simpy.Environment, factory:SimpleFactory, action:int):
    """
    produce components using resources
    """
    if action==0: #use resources to produce c1
        yield factory.resources.get(2) # use 2 units of resources to produce 1 unit of c1
        produce_time = max(random.gauss(3, 0.3),2.5)
        yield env.timeout(produce_time)
        yield factory.c1.put(1) #produce 1 unit of component c1
        
    elif action==1: #use resources to produce c2
        yield factory.resources.get(1)  # use 1 unit of resource to produce 1 unit of c2
        produce_time = max(random.gauss(1,0.1), 0.8)
        yield env.timeout(produce_time)
        yield factory.c2.put(1)   
    else:
        yield env.timeout(0)



def assemble(env:simpy.Environment, factory:SimpleFactory):
    """
    assemble 1 unit of product using 1 unit of c1 and 3 units of c2
    """
    assemble_time = max(random.gauss(2,0.2),1.8)
    if factory.c1.level >= 1 and factory.c2.level >=3:
        yield factory.c1.get(1)
        yield factory.c2.get(3)
        yield env.timeout(assemble_time)
        yield factory.products.put(1)
        return 1 #number of products
    else:
        return 0 #no products