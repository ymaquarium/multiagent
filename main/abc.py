import numpy as np 
import matplotlib.pyplot as plt
import random
import sys, os

#explain usage
if len(sys.argv) < 3:
    print("Usage: python3 abc.py dimension epoch sample save_dir_name")
    sys.exit()

#general 
N = int(sys.argv[1])
epoch = int(sys.argv[2])
sample = int(sys.argv[3])
save_dir =  "../results/" + sys.argv[4] + "/"
os.makedirs(save_dir, exist_ok=True)

#functions
func_list = ["Sphere", "Rastrign", "Rosenbrock", "Griewank", "Alpine", "Two_N_minima"]

def Sphere(array, N):
    return np.sum(np.square(array))

def Rastrign(array, N):
    return 10*N + np.sum(np.square(array)-10*np.cos(2*np.pi*array))

def Rosenbrock(array, N):
    s = 0
    for i in range(N-1):
        s += 100*np.square(array[i+1]-np.square(array[i])) +np.square(array[i])
    return s

def Griewank(array, N):
    return 1+np.sum(np.square(array))/4000-np.prod(np.cos(array/np.sqrt(np.arange(1, N+1))))

def Alpine(array, N):
    return np.sum(np.abs(array*np.sin(array)+0.1*array))

def Two_N_minima(array, N):
    return np.sum(np.square(np.square(array))-16*np.square(array)+5*array)

#for evaluate fitness in onlooker_bee
def softmax(N, sample, array, func_name):
    M = max([func_name(array[i], N) for i in range(sample)])
    eachArray = np.arange(sample, dtype="float")
    for i in range(sample):
        eachArray[i] = np.exp(func_name(array[i], N)-M)
    sum_exp = np.sum(eachArray)
    y = eachArray/sum_exp

    return y 

#update 
def update_honey(honey1, honey2, func_name, trial_count_i):
    r = np.diag(np.random.rand(N)*2.0-1.0)
    honey_cande = honey1 +  np.dot(r, honey1-honey2)
    if func_name(honey_cande, N) < func_name(honey1, N):
        honey1 = honey_cande
        trial_count_i=0
    else:
        trial_count_i+=1
    
    return func_name(honey1, N), honey1

def employed_bee(N, sample, honey_source, func_name, trial_count):
    honey_value = []
    honey_point = []
    for i in range(sample):
        k = i 
        while True:
            k = np.random.choice(np.arange(sample))
            if k != i:
                break
        local_honey, honey = update_honey(honey_source[i], honey_source[k], func_name, trial_count[i])
        honey_point.append(honey)
        honey_value.append(local_honey)
    
    return [min(honey_value), honey_point[honey_value.index(min(honey_value))]]

def onlooker_bee(N, sample, honey_source, func_name, trial_count):
    select_prob = softmax(N, sample, honey_source, func_name)
    honey_value = []
    honey_point = []

    for i in range(sample):
        selected_index = np.random.choice(np.arange(sample), p=select_prob.tolist())
        k = i 
        while True:
            k = np.random.choice(np.arange(sample))
            if k != selected_index:
                break
        local_honey, honey = update_honey(honey_source[selected_index], honey_source[k], func_name, trial_count[selected_index])
        honey_point.append(honey)
        honey_value.append(local_honey)
    
    return [min(honey_value), honey_point[honey_value.index(min(honey_value))]]

#trial_limit = N*sample
#https://www.jstage.jst.go.jp/article/jceeek/2010/0/2010_0_267/_pdf

def scout_bee(N, sample, honey_source, x_min, x_max, func_name, trial_count):
    honey_value = []
    honey_point = []
    for i in range(sample):
        if trial_count[i] >= N*sample:
            honey_source[i] = (x_max-x_min)*np.random.rand(N)+x_min
            trial_count[i] = 0
        honey_value.append(func_name(honey_source[i], N))
        honey_point.append(honey_source[i])
    
    return [min(honey_value), honey_source[honey_value.index(min(honey_value))]]

def ABC(N, sample, func_name, x_min, x_max, epoch):
    honey_source = [(x_max-x_min)*np.random.rand(N)+x_min for i in range(sample)]
    trial_count = np.zeros(sample)
    best_honey_point = np.zeros(N)
    best_honey = np.inf

    #graph
    score_trend = [best_honey]

    for _ in range(epoch):
        comp_honey = [[best_honey, best_honey_point]]
        comp_honey.append(employed_bee(N, sample, honey_source, func_name, trial_count))
        comp_honey.append(onlooker_bee(N, sample, honey_source, func_name, trial_count))
        comp_honey.append(scout_bee(N, sample, honey_source, x_min, x_max, func_name, trial_count))
        best_honey, best_honey_point = sorted(comp_honey)[0]

        if _%100==0:
            score_trend.append(best_honey)
    
    return np.array(score_trend), best_honey_point

#optimize functions by ABC
def executer(N, sample, func_name, epoch):
    if func_name == "Sphere":
        return ABC(N, sample, Sphere, -5.0, 5.0, epoch)
    elif func_name == "Rastrign":
        return ABC(N, sample, Rastrign, -5.0,  5.0, epoch)
    elif func_name == "Rosenbrock":
        return ABC(N, sample, Rosenbrock, -5.0, 10.0, epoch)
    elif func_name == "Griewank":
        return ABC(N, sample, Griewank, -600.0, 600.0, epoch)
    elif func_name == "Alpine":
        return ABC(N, sample, Alpine, -10.0, 10.0, epoch)
    elif func_name == "Two_N_minima":
        return ABC(N, sample, Two_N_minima, -5.0, 5.0, epoch)

#generate optimizing graph and optimizing point
def result_generator(func_list, N, sample, epoch):
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for i in range(len(func_list)):
        x = np.arange(epoch/100+1)
        y, point = executer(N, sample, func_list[i], epoch)

        #optimizing point
        np.set_printoptions(formatter={"float": '{:0.15f}'.format}) 
        f = open(save_dir + "ABC Optimizing Point of "+func_list[i]+".txt", "w")
        f.write(str(point))
        f.close

        #graph paint
        plt.subplot(3, 2, i+1)
        plt.plot(x, y, color='red', linestyle='solid', linewidth=1.0)
        plt.title(func_list[i] + " Optimized by ABC")
        plt.xlim((-1, epoch/100+1))
        plt.xlabel("Update Count")
        plt.ylabel("Value of "+func_list[i]+" Function")
        if np.all(y>0):
            plt.yscale('log')
        plt.grid(True)
    plt.savefig(save_dir + 'ABC.png')

if __name__ == "__main__":
    result_generator(func_list, N, sample, epoch)

