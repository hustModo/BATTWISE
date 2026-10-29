import heapq

def population_management(pop,size):
    pop = [individual for individual in pop if individual['objective'] is not None]
    if size > len(pop):
        size = len(pop)
    
    # 注释掉去重逻辑，保持1:1对应
    # unique_pop = [] 
    # unique_objectives = []
    # for individual in pop:
    #     if individual['objective'] not in unique_objectives:
    #         unique_pop.append(individual)
    #         unique_objectives.append(individual['objective'])
    
    # 直接选择最优的size个个体，不去重
    pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    return pop_new