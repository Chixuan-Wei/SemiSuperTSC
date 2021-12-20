from functions import *
from weinet import *
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from net_para import *
import argparse
import logging
import random
# os.system("nvidia-smi")
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # set device
USE_CUDA = torch.cuda.is_available()
DEVICE = device
device = DEVICE
print("CUDA:", USE_CUDA, DEVICE, flush=True)
dataset_path = os.path.join(os.getcwd(), 'UCRArchive_2018')  # set dataset path
batch_code = "batch2"

percentage_subsequence_length = 0.3  # size of subsequence or window
percentage_stride = 0.05  # size of stride
level = 2  # [2,3,4,5,6,7,8,9,10]
num_epochs = 200  # epoch number
batch_size = 30  # batchsize
alpha = 0.3#[0,0.2,0.3,0.5]
lr = 0.0001
wd = 0.4
dataset_list = [
                    "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ", "ArrowHead",
                    "Beef", "BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown", "ChlorineConcentration",
                    "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ", "Crop",
                    "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
                    "DistalPhalanxTW", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "Earthquakes", "ECG200",
                    "ECG5000", "ECGFiveDays", "ElectricDevices", "EOGHorizontalSignal", "EOGVerticalSignal",
                    "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish",
                    "FordA", "FordB",
                    "FreezerRegularTrain", "FreezerSmallTrain", "Fungi",
                    "GestureMidAirD1", "GestureMidAirD2",
                    "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2", "GunPoint", "GunPointAgeSpan",
                    "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham", "HandOutlines", "Haptics", "Herring",
                    "HouseTwenty", "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectWingbeatSound",
                    "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat",
                    "MedicalImages", "MelbournePedestrian", "MiddlePhalanxOutlineAgeGroup",
                    "MiddlePhalanxOutlineCorrect",
                    "MiddlePhalanxTW", "MixedShapesRegularTrain", "MixedShapesSmallTrain", "MoteStrain",
                    "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf",
                    "PhalangesOutlinesCorrect", "Phoneme", "PickupGestureWiimoteZ", "Pictures", "PigAirwayPressure",
                    "PigArtPressure", "PigCVP", "PLAID", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup",
                    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "Rock", "ScreenType",
                    "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2", "ShakeGestureWiimoteZ",
                    "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SmoothSubspace", "SonyAIBORobotSurface1",
                    "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
                    "SyntheticControl",
                    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UMD",
                    "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
                    "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"]

network_args = {
    "stride": percentage_stride,
    "horizon": percentage_subsequence_length,
    "level": level,
    "alpha": alpha,
    "run_times": "t3",
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "net_base": "wave-WeiUCR-UCR-lr{}-wd{}-bs{}".format(
        str(lr).replace('.', '_'), str(wd).replace('.', '_'), str(batch_size))
}



def run_network(dataset_name, net_info):
    data = loadDataset(dataset_path, dataset_name)

    # acquire the number of class
    num_classes = len(np.unique(data.values[:, 0]))

    # split train test dataset
    train_labeled_x, train_labeled_y, train_unlabeled_x, validate_x, validate_y, test_x, test_y = stratifiedSampling(
        data, seed=0, normalization=True, device=device)

    # calculate the length of stride
    sub_length = int(train_labeled_x.shape[2] * net_info.horizon)
    stride = int(train_labeled_x.shape[2] * net_info.stride)

    # get the subsequence
    pre, post = getPreAndPostSubsequences(torch.cat((train_labeled_x, train_unlabeled_x)), sub_length, stride)

    # calculate coefficient
    pre_coeffs = waveletTransform(pre.cpu(), level=net_info.level)
    pre_coeffs = pre_coeffs.to(device)
    post_coeffs = waveletTransform(post.cpu(), level=net_info.level)
    post_coeffs = post_coeffs.to(device)
    print(
        "horizon:{} stride{} level{} alpha{}".format(net_info.horizon, net_info.stride, net_info.level, net_info.alpha))


    # add logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # log_path = os.path.join('/data/home/u20120380', 'log','Weinetlog'+batch_code)
    log_path = os.path.join(os.getcwd(), 'log', 'SemiLog')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = os.path.join(log_path, dataset_name+rq+'.log')
    logfile=log_name
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # pre_subsequence predict post_subsequence
    subsequences = torch.cat((pre, post), dim=1)
    subsequences_coeffs = torch.cat((pre_coeffs, post_coeffs), dim=1)

    # build dataloader
    subsequences_iterate = DataLoader(subsequences, batch_size=batch_size)
    subsequences_coeffs_iterate = DataLoader(subsequences_coeffs, batch_size=batch_size)

    # create model
    alexnet_mod = ALEXNetMOD(int(pre.shape[2]), int(pre_coeffs.shape[2]), num_classes).to(device)
    #print(alexnet_mod)

    # loss function
    criterion_classification = nn.CrossEntropyLoss()  
    criterion_forecasting = nn.MSELoss()  

    # create optimizer
    optimizer = torch.optim.AdamW(alexnet_mod.parameters(), lr=lr, weight_decay=wd)

    # record the process of training
    dataset_stat = pd.DataFrame(
        columns=['epoch', 'train_acc', 'val_acc', 'test_acc', 'classification_loss', 'forecast_loss', 'time_cost'])
    best_accuracy = 0
    # train model
    for t in range(net_info.num_epochs):
        time_epoch_start = time.time()
        for batch, (subsequences_batch, subsequences_coeffs_batch) in enumerate(zip(subsequences_iterate,
                                                                                    subsequences_coeffs_iterate)):
            pre_subsequences_batch = subsequences_batch[:, 0, :].unsqueeze(1)
            post_subsequences_batch = subsequences_batch[:, 1, :].unsqueeze(1)
            pre_subsequences_coeffs_batch = subsequences_coeffs_batch[:, 0, :].unsqueeze(1)
            post_subsequences_coeffs_batch = subsequences_coeffs_batch[:, 1, :].unsqueeze(1)

            loss_classification, loss_forecast = optimize_network(alexnet_mod, optimizer, train_labeled_x, train_labeled_y,
                                                                  pre_subsequences_batch,
                                                                  post_subsequences_batch,
                                                                  pre_subsequences_coeffs_batch,
                                                                  post_subsequences_coeffs_batch, net_info.alpha)
        train_acc = accuracy_score(
            np.argmax(alexnet_mod.forward_test(train_labeled_x.float()).cpu().detach().numpy(), axis=1),
            train_labeled_y.long().cpu().numpy())  
        val_acc = accuracy_score(np.argmax(alexnet_mod.forward_test(validate_x.float()).cpu().detach().numpy(), axis=1),
                                 validate_y.long().cpu().numpy())
        time_epoch_end = time.time()
        epoch_time_cost = time_epoch_end - time_epoch_start
        print("one epoch time cost: {} s".format(epoch_time_cost))

        test_acc = accuracy_score(
            np.argmax(alexnet_mod.forward_test(test_x.float()).cpu().detach().numpy(), axis=1),
            test_y.long().cpu().numpy())

        current_accuracy = val_acc
        model_name = dataset_name+"_a_"+str(network_args["alpha"]).replace(".","")+"_"+"{}".format(str(val_acc).replace(".","")+"_"+".pth")
        net_info.model_save_pos =os.path.join(os.getcwd(), "model_saved",model_name )
        # torch.save(alexnet_mod.state_dict(), net_info.model_save_pos)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(alexnet_mod.state_dict(), net_info.model_save_pos)
        print(
            f'Epoch{t}   train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc}, classification_loss: {loss_classification}, prediction_loss: {loss_forecast}, time_cost{epoch_time_cost}')
        logger.info(
            f'Epoch{t}   train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc}, classification_loss: {loss_classification}, prediction_loss: {loss_forecast}, time_cost{epoch_time_cost}')
        # ['epoch', 'train_acc', 'val_acc', 'test_acc', 'avg loss', 'classification_loss', 'forecast_loss']

        dataset_stat = dataset_stat.append(
            {'epoch': t, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
             'classification_loss': loss_classification, 'forecast_loss': loss_forecast, 'time_cost':epoch_time_cost},
            ignore_index=True
        )

    logger.info(
        f'Best_accuracy:{best_accuracy}')
    dataset_stat = dataset_stat.append(
        {'epoch': 'best_accuracy:{}'.format(best_accuracy), 'train_acc': net_info.model_save_pos}, ignore_index=True)
    dataset_stat.to_csv(net_info.dataset_stat_pos)
    return best_accuracy


def one_run(horizon=network_args["horizon"], stride=network_args["stride"],
            level=network_args["level"], alpha=network_args["alpha"], run_time=network_args["run_times"],
            dataset_list=dataset_list):
    network_args["horizon"] = horizon
    network_args["stride"] = stride
    network_args["level"] = level
    network_args["alpha"] = alpha
    network_args["run_times"] = run_time

    # try to use net_info to manage the parameters
    net_info = net_para()

    datasets_stat = pd.DataFrame(columns=['Dataset', 'Accuracy', 'horizon', 'stride', 'level', 'alpha'])
    # print(net_info.onerun_suffix)
    for dataset in dataset_list:
        if not net_info.init_network(parser=argparse.ArgumentParser(), network_args=network_args):
            print("{} has problem.".format(net_info.onerun_suffix))
            return False
        #        net_info.current_dataset = dataset
        print(net_info.onerun_suffix)
        print(dataset)
        net_info.dataset_run_set(dataset)
        if os.path.exists(net_info.dataset_stat_pos):
            print("{} has been processed".format(net_info.dataset_stat_pos))
            continue

        try:
            # train model
            time_netrun_start = time.time()
            data_set_accuracy = run_network(dataset, net_info)
            time_netrun_end = time.time()
            print("total time cost {} s".format(time_netrun_end - time_netrun_start))
            datasets_stat = datasets_stat.append({'Dataset': dataset, 'Accuracy': data_set_accuracy, 'horizon': net_info.horizon,
                                  'stride': net_info.stride, 'level': net_info.level, 'alpha': net_info.alpha},
                                 ignore_index=True
                                 )
            datasets_stat.to_csv(net_info.datasets_stat_pos)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('\n' + message)
            continue

    return True

