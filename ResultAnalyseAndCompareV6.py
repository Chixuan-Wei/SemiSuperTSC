import pandas as pd
import numpy as np
import os
import time

timestamp_iden = str(time.time()).split('.')[0]
PARALEN = 4
'''
used to get all result
'''

def check_dir_exists(dir):
    if not os.path.exists():
        os.mkdir(dir)


def get_sub_dirlist(dir_path):
    dir_list = os.listdir(dir_path)

    dir_list = list(filter(lambda x: os.path.isdir(os.path.join(dir_path, x)), dir_list))
    dir_list = list(filter(lambda x: len(x.split("_")) >= PARALEN, dir_list))
    #  print(dir_list)
    return dir_list


def get_csv_files(dir_path):
    file_list = os.listdir(dir_path)
    # get all csv file
    csv_list = filter(lambda x: '.csv' in x and 'onerun' not in x, file_list)
    return list(csv_list)


def get_parameter(csv_name):
    name_split = csv_name.split("_")
    dataset = name_split[0]
    horizon = name_split[name_split.index("h") + 1]
    stride = name_split[name_split.index("s") + 1]
    alpha = name_split[name_split.index("a") + 1]
    level = name_split[name_split.index("l") + 1].split('.')[0]
    parameter_dict = {
        "dataset": dataset,
        "horizon": horizon,
        "stride": stride,
        "alpha": alpha,
        "level": level
    }
    return parameter_dict


def analyse_csv(csv_pos, csv_name):
    csv_para = get_parameter(csv_name)
    print(csv_pos)

    try:
        df = pd.read_csv(csv_pos)
    except pd.errors.EmptyDataError:
        print(" is empty".format(csv_pos))
        os.remove(csv_pos)
        return csv_para, 0, 0, 0
    # print(df["test_acc"])
    # print(df.loc[df.shape[0]-1])
    df = df.drop(df.shape[0] - 1)
    # print(df.loc[df.shape[0] - 1])
    best_test_accuracy = df["test_acc"].max()
    best_test_epoch = df[df["test_acc"] == best_test_accuracy]["epoch"].to_list()
    # print(df["val_acc"])
    best_val_accuracy = df["val_acc"].max()
    best_val_epoch = df[df["val_acc"] == best_val_accuracy]["epoch"].to_list()
    best_val_test_acc = df[df["val_acc"] == best_val_accuracy]["test_acc"].to_list()

    best_train_accuracy = df["train_acc"].max()
    best_train_epoch = df[df["train_acc"] == best_train_accuracy]["epoch"].to_list()
    return csv_para, best_test_accuracy, best_val_accuracy, best_val_test_acc , best_train_accuracy, best_test_epoch, best_val_epoch, best_train_epoch


def summary_nets():
    summary_of_nets_dir = os.path.join(os.getcwd(), "summary_of_nets")
    check_dir_exists(summary_of_nets_dir)
    summary_of_nets_file = "summary_of_nets_" + timestamp_iden + ".csv"
    summary_of_nets_pos = os.path.join(summary_of_nets_dir, summary_of_nets_file)


# 针对各个数据集分开统计
def separete_stat(total_df, net_res_dir, net="wave-alex"):
    dataset_list = total_df["dataset"].unique()
    total_summary_df = pd.DataFrame()

    for dataset in dataset_list:
        dataset_stat_file = dataset + ".csv"
        dataset_stat_pos = os.path.join(net_res_dir, dataset_stat_file)
        dataset_stat_df = total_df[total_df["dataset"] == dataset]
        dataset_stat_df.to_csv(dataset_stat_pos)
        dataset_best_accuracy = dataset_stat_df["best_test_accuracy"].max()
        total_summary_df = pd.concat(
            [total_summary_df, dataset_stat_df[dataset_stat_df["best_test_accuracy"] == dataset_best_accuracy]], axis=0)

    total_summary_file = "summary_" + net + "_.csv"
    total_summary_pos = os.path.join(net_res_dir, total_summary_file)
    total_summary_df.to_csv(total_summary_pos)


def analyse_net(net):
    datasets_list = []

    net_res_dir = os.path.join(os.getcwd(), "result",  net)
    result_dir_list = get_sub_dirlist(net_res_dir)

    net_all_res_df = pd.DataFrame()
    net_all_res_df_file = "all_res.csv"
    net_all_res_df_pos = os.path.join(net_res_dir, net_all_res_df_file)

    # 同一net下不同参数的结果
    for result_dir_name in result_dir_list:
        result_dir_pos = os.path.join(net_res_dir, result_dir_name)
        #   print(result_dir_name)
        csv_list = get_csv_files(result_dir_pos)
        if len(csv_list) == 0:
            continue
        onerun_summary_df = pd.DataFrame(columns=
                                         ['dataset', 'best_test_accuracy', "best_test_epoch", 'best_val_accuracy',
                                           'best_val_epoch' ,'best_val_test_acc', 'best_train_accuracy', 'best_train_epoch',
                                          'horizon', 'stride', 'level', 'alpha']
                                         )

        for csv_file in csv_list:
            print(csv_file)
            csv_pos = os.path.join(result_dir_pos, csv_file)
            ################ print(csv_pos)
            para_list, best_test_accuracy, best_val_accuracy,best_val_test_acc, best_train_accuracy, best_test_epoch, best_val_epoch, best_train_epoch = analyse_csv(
                csv_pos, csv_file)
            if best_test_accuracy == 0 and best_val_accuracy == 0:
                continue
            datasets_list.append(para_list["dataset"])
            onerun_summary_df = onerun_summary_df.append(
                {'dataset': para_list["dataset"], 'best_test_accuracy': best_test_accuracy,
                 'best_test_epoch': best_test_epoch
                    , 'best_val_accuracy': best_val_accuracy, 'best_val_epoch': best_val_epoch,
                 'best_val_test_acc':best_val_test_acc,
                 'best_train_accuracy': best_train_accuracy
                    , 'best_train_epoch': best_train_epoch, 'horizon': para_list["horizon"],
                 'stride': para_list['stride'], 'level': para_list['level'], 'alpha': para_list['alpha']
                 },
                ignore_index=True
            )

        onerun_summary_file_list = csv_list[0].split('_')
        onerun_summary_file_list[0] = "onerun"

        onerun_summary_file = '_'.join(onerun_summary_file_list)
        onerun_summary_pos = os.path.join(result_dir_pos, onerun_summary_file)
        print(onerun_summary_pos)
        onerun_summary_df.to_csv(onerun_summary_pos)
        net_all_res_df = pd.concat([net_all_res_df, onerun_summary_df], axis=0)
    net_all_res_df.to_csv(net_all_res_df_pos)
    separete_stat(net_all_res_df, net_res_dir, net)
    print(datasets_list)


def analyse_nets(nets_list):
    for net in nets_list:
        print(net)
        analyse_net(net)


if __name__ == "__main__":

    nets_list = [
        "wave-WeiUCR-UCR-lr0_0001-wd0_4-bs30_"
                 ]

    analyse_nets(nets_list)
