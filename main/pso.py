import numpy as np
import matplotlib.pyplot as plt  
import random
import sys, os

#explain usage
if len(sys.argv) < 4:
    print("Usage: python3 pso.py dimension epoch sample save_dir_name")

#general 
N = int(sys.argv[1])
epoch = int(sys.argv[2])
sample = int(sys.argv[3])
save_dir = "../results/" + sys.argv[4] + "/"
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
        s += 100*np.square(array[i+1]-np.square(array[i])) +np.square(array[i]-1)
    return s 

def Griewank(array, N):
    return 1+np.sum(np.square(array))/4000-np.prod(np.cos(array/np.sqrt(np.arange(1, N+1))))

def Alpine(array, N):
    return np.sum(np.abs(array*np.sin(array)+0.1*array))

def Two_N_minima(array, N):
    return np.sum(np.square(np.square(array))-16*np.square(array)+5*array)


def PSO(N, sample, func_name, x_min, x_max, epoch):
    
    r1 = random.random()
    r2 = random.random()

    #best_score and best_points
    global_best_score = np.inf 
    global_best_point = np.zeros(N)
    local_best_score = [np.inf for i in range(sample)]
    local_best_point = [np.zeros(N) for i in range(sample)]

    #particles
    velocity = [np.zeros(N) for i in range(sample)]
    current_x = [(x_max-x_min)*np.random.rand(N)+x_min for i in range(sample)]

    #graph
    score_trend = []

    #first evaluation
    for i in range(sample):
        value = func_name(current_x[i], N)
        if value < local_best_score[i]:
            local_best_score[i] = value
            local_best_point[i] = current_x[i] 

        #check
        #print(str(func_name))
        #print(local_best_score)
    global_best_score = min(local_best_score)
    global_best_point = local_best_point[local_best_score.index(min(local_best_score))]
    score_trend.append(global_best_score)

    #coefficients
    c1 = 1.8
    c2 = 0.2
    w = 0.8
    #https://www.jstage.jst.go.jp/article/jasmin/2010f/0/2010f_0_84/_pdf/-char/ja

    for time in range(epoch):

        #parameter tuning
        if time >= 2*epoch//3:
            c1, c2 = 0.2, 1.8
        elif time >= epoch//3:
            c1, c2 = 1.0, 1.0

        #update
        for i in range(sample):
            velocity[i] = w*velocity[i] + c1*r1*(local_best_point[i]-current_x[i]) + c2*r2*(global_best_point-current_x[i])
            current_x[i] = current_x[i] + velocity[i]
    
        #evaluation
        for i in range(sample):
            value = func_name(current_x[i], N)
            if value < local_best_score[i]:
                local_best_score[i] = value 
                local_best_point[i] = current_x[i]
            #check 
            #if time%100==0:
                #print(str(func_name))
                #print(local_best_score)
        global_best_score = min(local_best_score)
        global_best_point = local_best_point[local_best_score.index(min(local_best_score))]
        if time%100 == 0:
            score_trend.append(global_best_score)

    return (np.array(score_trend), global_best_point)

#optimize every function by PSO
def executer(N, sample, func_name, epoch):
    if func_name == "Sphere":
        return PSO(N, sample, Sphere, -5.0, 5.0, epoch)
    elif func_name == "Rastrign":
        return PSO(N, sample, Rastrign, -5.0,  5.0, epoch)
    elif func_name == "Rosenbrock":
        return PSO(N, sample, Rosenbrock, -5.0, 10.0, epoch)
    elif func_name == "Griewank":
        return PSO(N, sample, Griewank, -600.0, 600.0, epoch)
    elif func_name == "Alpine":
        return PSO(N, sample, Alpine, -10.0, 10.0, epoch)
    elif func_name == "Two_N_minima":
        return PSO(N, sample, Two_N_minima, -5.0, 5.0, epoch)

#generate optimizing graph and optimizing point
def result_generator(func_list, N, sample, epoch):
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for i in range(len(func_list)):
        x = np.arange(epoch/100+1)
        y, point = executer(N, sample, func_list[i], epoch)

        #optimizing point
        np.set_printoptions(formatter={"float": '{:0.15f}'.format}) 
        f = open(save_dir + "PSO Optimizing Point of "+func_list[i]+".txt", "w")
        f.write(str(point))
        f.close

        #graph paint
        plt.subplot(3, 2, i+1)
        plt.plot(x, y, color='red', linestyle='solid', linewidth=1.0)
        plt.title(func_list[i] + " Optimized by PSO")
        plt.xlim((-1, epoch/100+1))
        plt.xlabel("Update Count")
        plt.ylabel("Value of "+func_list[i]+" Function")
        if np.all(y>0):
            plt.yscale('log')
        plt.grid(True)
    plt.savefig(save_dir + 'PSO.png')
     

if __name__ == "__main__":
    result_generator(func_list, N, sample, epoch)


