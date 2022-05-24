#!/usr/bin/env python3

import os
import shutil
import xlsxwriter


""""IMPORTANT NOTICE: depending on the testing and the option the main function arguments needs to be configured properly
THATS depends on the testing which at the moment is on board. FOR EXAMPLE:
os.system('python main.py 700 0.2 1 1 3 200 0.091') would result in an error because the current main function is configured on"""

#creates the initial structure of the model
core_path = "C:/Users/nrk_pavilion/PycharmProjects/WGSB_1"

def init_function(test_path):
    #create txtfiles: the collection storage directory
    temp = [test_path, "/txtfiles"]
    path = "".join(temp)
    try:
        os.mkdir(path)
    except:
        print("EXISTS txtfiles")
    #create figures: the results files
    temp =[test_path, "/figures"]
    path = "".join(temp)
    try:
        os.mkdir(path)
    except:
        print("EXISTS figures")
    temp = [test_path, "/figures/CF"]
    path = "".join(temp)
    try:
        os.mkdir(path)
    except:
        print("EXISTS CF")
    temp = [test_path, "/figures/CRAN"]
    path = "".join(temp)
    try:
        os.mkdir(path)
    except:
        print("EXISTS CRAN")
    temp = [test_path, "/figures/TIME"]
    path = "".join(temp)
    try:
        os.mkdir(path)
    except:
        print("EXISTS TIME")
    temp = [test_path, "/figures/NPL"]
    path = "".join(temp)
    try:
        os.mkdir(path)
    except:
        print("EXISTS NPL")
    #create backup: usefull directory to store items collection not code important though
    temp=[test_path, "/backup"]
    path = "".join(temp)
    try:
        os.mkdir(path)
    except:
        print("EXISTS Backup")

    return 0

if "txtfiles" not in os.listdir(core_path):
    init_function(core_path)
    print("Init_process_starts")
else:
    print("TESTING!")

def PrepareForNextRound(counter,test_str,col):
    path = ["figures","/",col]
    path = "".join(path)
    namelist = [str(counter)," ", test_str," ",col]
    name = "".join(namelist)
    print(name)
    dirpath = os.path.join(path, name)
    print(dirpath)
    try:
        os.mkdir(dirpath)
    except:
        print("EXISTS")

    targets = ["DotSplit.dat","NegMain.dat","invertedindex.dat","PerSplit.dat",
               "docinfo.dat","ConstantWindFile.dat","SenParConWind.dat",
                "densfile.dat","newfile.dat","debuglog.dat","invertedindex.dat","CoreRankfile.dat","new_res.xlsx"]
    for file in os.listdir(core_path):
        if file in targets:
            try:
                shutil.copy2(file, dirpath)
            except FileNotFoundError:
                print(file + "NOT EXists")
            try:
                os.remove(file)
            except FileNotFoundError:
                print(file + "NOT EXists to delete")

    workbook = xlsxwriter.Workbook('new_res.xlsx')
    workbook.close()
    print("Copy paste your PARSED collection in TXTFILES and start over")
    exit(0)


counter = 0

#name of main: sys.args[1] =  h - sys.args[2] =  p - sys.args[3] = menu 1 - sys.args[4] = choice of menu 1 -
#sys.args[5] = constant window size, sys.args[6] = paragraph size - sys.args[7] = percentage window.
#example:  main_v2.py 700 0.2 1 1 3 200 0.091')

#os.system( "C:/Users/nrk_pavilion/AppData/Local/Programs/Python/Python38/python.exe")

# test with window size 3
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 1 3 80 0.091') # maincore
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 2 3 80 0.091') # GSB
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 3 3 80 0.091') # Density
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 4 3 80 0.091') # CoreRank
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 5 3 80 0.091') # COnstant
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 6 3 80 0.091') # SenPar
#
#
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.2 2')

#name for file in figures directory
testname = "GSB_w_cores_constant"
col = "CF"

PrepareForNextRound(counter,testname,col)
counter+=1

# test with window size 5
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 1 5 80 0.091') # maincore
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 2 5 80 0.091') # GSB
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 3 5 80 0.091') # Density
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 4 5 80 0.091') # CoreRank
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 5 5 80 0.091') # COnstant
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 6 5 80 0.091') # SenPar
#
#
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.2 2')

#name for file in figures directory
testname = "GSB_w_cores_constant"

PrepareForNextRound(counter,testname,col)
counter+=1

# test with window size 8
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 1 8 80 0.091') # maincore
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 2 8 80 0.091') # GSB
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 3 8 80 0.091') # Density
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 4 8 80 0.091') # CoreRank
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 5 8 80 0.091') # COnstant
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 6 8 80 0.091') # SenPar
#
#
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.2 2')

#name for file in figures directory
testname = "GSB_w_cores_constant"

PrepareForNextRound(counter,testname,col)
counter+=1



# test with window size 11
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 1 11 80 0.091') # maincore
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 2 11 80 0.091') # GSB
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 3 11 80 0.091') # Density
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 4 11 80 0.091') # CoreRank
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 5 11 80 0.091') # COnstant
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 6 11 80 0.091') # SenPar
#
#
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.2 2')

#name for file in figures directory
testname = "GSB_w_cores_constant"

PrepareForNextRound(counter,testname,col)
counter+=1

# test with window size 14
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 1 14 80 0.091') # maincore
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 2 14 80 0.091') # GSB
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 3 14 80 0.091') # Density
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 4 14 80 0.091') # CoreRank
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 5 14 80 0.091') # COnstant
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 6 14 80 0.091') # SenPar
#
#
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.2 2')

#name for file in figures directory
testname = "GSB_w_cores_constant"

PrepareForNextRound(counter,testname,col)
counter+=1

# test with window size 17
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 1 17 80 0.091') # maincore
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 2 17 80 0.091') # GSB
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 3 17 80 0.091') # Density
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 4 17 80 0.091') # CoreRank
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 5 17 80 0.091') # COnstant
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 6 17 80 0.091') # SenPar
#
#
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.2 2')

#name for file in figures directory
testname = "GSB_w_cores_constant"

PrepareForNextRound(counter,testname,col)
counter+=1

# test with window size 20
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 1 20 80 0.091') # maincore
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 2 20 80 0.091') # GSB
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 3 20 80 0.091') # Density
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 4 20 80 0.091') # CoreRank
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 5 20 80 0.091') # COnstant
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.4 1 6 20 80 0.091') # SenPar
#
#
os.system('python main_test1_gsb_maincore_density_corerank_constant_senpar.py 700 0.2 2')

#name for file in figures directory
testname = "GSB_w_cores_constant"

PrepareForNextRound(counter,testname,col)
counter+=1