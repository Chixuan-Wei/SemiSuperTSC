import os
import time


class net_para():
    def __init__(self):
        self.model_save_pos = ""

        self.onerun_suffix = ""
        #     self.res_file_name=""
        self.res_save_dir = ""
        self.dataset_stat_filename = ""
        self.datasets_stat_filename = ""
        self.timestamp_iden = ""
        self.datasets_stat_pos = ""
        self.dataset_stat_pos = ""

        ## result for single dataset
        self.dataset_res_name = ""
        ## result for all dataset
        self.run_time_res = ""
        #  self.test_type="origin_split" #duplicated all use equal_split

        self.base_net = "Wave-Alex"
        self.net_type = ""
        self.run_times = "tmp"
        self.batch_size = 30

        self.horizon = 1
        self.stride = 1
        self.alpha = 1
        self.level = 1
        self.batch_size = 100
        self.num_epochs = 200
        self.net_base = ""
        self.regu_type = "L2"
        self.current_dataset = "tmp"

    def gen_suffix(self):
        self.onerun_suffix = '_h_' + str(self.horizon) + '_s_' + str(self.stride) + "_a_" + str(
            self.alpha) + "_l_" + str(self.level)

    # self.net_type=""
    def init_network(self, network_args, parser):
        #   parser.add_argument("--test_type", type=str, default='origin_split')
        parser.add_argument("--run_times", type=str, default=network_args["run_times"])
        parser.add_argument("--horizon", type=float, default=network_args["horizon"])
        parser.add_argument("--stride", type=float, default=network_args["stride"])
        parser.add_argument("--alpha", type=float, default=network_args["alpha"])
        parser.add_argument("--level", type=float, default=network_args["level"])
        args = parser.parse_args()

        self.run_times = args.run_times
        self.horizon = args.horizon
        self.stride = args.stride
        self.alpha = args.alpha
        self.level = args.level

        if self.horizon < self.stride or self.horizon+self.stride > 1:
            return False

        self.batch_size = network_args["batch_size"]
        self.num_epochs = network_args["num_epochs"]
        self.net_base = network_args["net_base"]


        network_args["run_times"] = args.run_times
        network_args["horizon"] = args.horizon
        network_args["stride"] = args.stride
        network_args["alpha"] = args.alpha
        network_args["level"] = args.level

        #  self.net_type = self.net_base + "_"
        self.set_net_type()
        self.gen_suffix()
        self.timestamp_iden = str(time.time()).split('.')[0]
        self.set_save_dirs()
        # self.model_save_pos = os.path.join(self.model_save_dir,
        #                                    self.net_type + "_" + self.current_dataset + "_" + self.timestamp_iden + ".pth")

        self.datasets_stat_filename = "onerun" + self.run_times + self.onerun_suffix + "_" + self.timestamp_iden + ".csv"
        self.datasets_stat_pos = os.path.join(self.res_save_dir, self.datasets_stat_filename)

        return network_args

    def set_net_type(self):
        self.net_type = self.net_base + "_"

    def set_save_dirs(self):
        ## single dataset dir:save test_df
        ## total result_dir: save result_df
        ## model save dir: save model
        ## if dir not exists, create it

        ## in case use colab
        # if train_env == 'colab':
        #     # 运行环境为线上时的配置
        #     data_path = os.getcwd() + '/drive/MyDrive/Datasets/UCRArchive_2018'd
        #     res_save_dir = os.getcwd() + '/drive/MyDrive/result/' + self.net_type + '/' + self.test_type + '_' + self.run_times + '/'
        # else:
        #     # 运行环境为线下时的配置
        #     data_path = os.getcwd() + "/../UCRArchive_2018"
        #     res_save_dir = os.getcwd() + '/../result/' + self.net_type + '/' + self.test_type + '_' + self.run_times + '/'

        self.res_save_dir = os.getcwd() + '/result/' + self.net_type + '/' + self.run_times + "_" + self.onerun_suffix + '/'

        if not os.path.exists(self.res_save_dir):
            os.makedirs(self.res_save_dir)

    # self.result_df_save_path = os.path.join(self.res_save_dir, self.result_df_filename)

    def get_datasets_stat_filename(self):
        self.datasets_stat_filename = self.run_times + self.onerun_suffix + "_" + self.timestamp_iden + ".csv"
        return self.datasets_stat_filename

    def dataset_run_set(self, dataset):
        self.current_dataset = dataset
        self.dataset_stat_filename = self.current_dataset + self.onerun_suffix + ".csv"
        self.dataset_stat_pos = os.path.join(self.res_save_dir, self.dataset_stat_filename)

        self.model_save_dir = os.path.join(os.getcwd(), 'model_saved')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
