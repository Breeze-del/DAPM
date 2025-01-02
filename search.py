from multiprocessing import Pool
#Grid Search code

import os
import sys
import time



def run_model(alpha, dropout_ration, weight_negative, dataset, gpu,weight_1, weight_2, weight_3):
    cmd = "python run_full.py"
    set = cmd + " --alpha " +str(alpha)+ " --dropout_ration " + str(dropout_ration) + \
    " --weight_negative " +str(weight_negative) + " --dataset " +str(dataset) +" --gpu " +str(gpu)  +" --weight_1 " +str(weight_1) + " --weight_2 " +str(weight_2) +" --weight_3 " +str(weight_3)
    print(set)
    os.system(set)
    
#" --code-length 32 --prun " + "16,8,4 " + \


if __name__ == '__main__':
    ind = int(sys.argv[0]) #which dataset to train 0/1
    gpu = sys.argv[0] # which gpu to use
    # beta1_list = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    if ind == 0:
        dropout_ration_list = [0.2]
        weight_negative_list = [0.1]
        alpha_list = [0.1]
        weight_1_list = [1/6]
        weight_2_list = [2/3]
    else:
        dropout_ration_list = [0.5]
        weight_negative_list = [0.01]
        alpha_list = [0.5]
        weight_1_list = [1/6]
        weight_2_list = [2/3]
    
    for alpha in alpha_list:
        for dropout_ration in dropout_ration_list:
            for weight_negative in weight_negative_list:
                for weight_1 in weight_1_list:
                    for weight_2 in weight_2_list:
                                    
                        time.sleep(2)
                        weight_3 = 1-weight_1 -weight_2
                        run_model(alpha, dropout_ration, weight_negative, ind, gpu,weight_1, weight_2, weight_3)
