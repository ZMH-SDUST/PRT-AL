# -*- coding: utf-8 -*-
"""
@Time ： 2022/5/5 11:16
@Auther ： Zzou
@File ：main.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import torch.optim as optim
import sys
import time
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from Tools.logger import Logger
from pytorchtools import EarlyStopping
from processtransformer import *
from models import *
from Dataloader import *
from RTF_Transformer import *
from src.Biatt_qrnn import qrnn7
from src.CRNN_att import BiCRNNAtt

warnings.filterwarnings("ignore")


def setup_seed(seed):
    np.random.seed(seed)  # numpy
    random.seed(seed)  # tensor
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # parallel gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu results are consistent
    torch.backends.cudnn.benchmark = True  # speed up training when the training set doesn't change much


class ScheduledOptim():
    """A simple wrapper class for learning rate scheduling"""
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def train_epoch(model, training_data, optimizer, loss_function, device):
    total_loss = 0
    batch_num = 0
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        batch_num += 1
        (x_train, y_train) = batch
        x_train = x_train.to(device)
        y_train = y_train.unsqueeze(1).to(device)
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_function(pred.float(), y_train.float())
        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()
    avg_loss = total_loss / batch_num
    return avg_loss


def eval_epoch(model, validation_data, loss_function, device, y_test_predict_true_file, eval_type):
    total_loss = 0
    batch_num = 0
    desc = '  - (Validation) '
    if eval_type == "test":
        with open(y_test_predict_true_file, "w", encoding='utf-8') as f:
            f.write("y_prediction" + " " + "y_true\n")
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            batch_num += 1
            (x_test, y_test) = batch
            x_test = x_test.to(device)
            y_test = y_test.unsqueeze(1).to(device)
            pred = model(x_test)
            loss = loss_function(pred.float(), y_test.float())
            total_loss += loss.item()
            if eval_type == "test":
                pred = pred.cpu().numpy()
                y_test = y_test.cpu().numpy()
                with open(y_test_predict_true_file, "a", encoding='utf-8') as f:
                    for i in range(len(pred)):
                        f.write(str(pred[i][0]) + " " + str(y_test[i][0]) + "\n")
    print(" eval avg loss is :", total_loss / batch_num)
    if eval_type == "train":
        return total_loss / batch_num


def padding(data, max_len, feature_dim):
    filler = [0.0 for i in range(feature_dim)]
    for i in range(len(data)):
        len_X_i = len(data[i])
        new_data = list()
        for j in range(max_len - len_X_i):
            new_data.append(filler)
        for k in range(len_X_i):
            new_data.append(data[i][k])
        data[i] = new_data
    return np.array(data)


if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='cuda or cpu', type=str, default='cuda', required=False)
    parser.add_argument('--epochs', help='training epochs', type=int, default=150, required=False)
    parser.add_argument('--mode', help='train or test', type=str, default='Train', required=False)
    parser.add_argument('--model', help='Model name', type=str, default='rtf_transformer', required=False)
    parser.add_argument('--rtf_regressor', help='regressor of  rtf_transformer', type=str,
                        default='Regressor_TemporalBlock', required=False)
    parser.add_argument('--bi', help='bidirectional', type=bool, default=True, required=False)
    parser.add_argument('--att', help='attention', type=bool, default=True, required=False)
    parser.add_argument('--node', help='rnn node', type=str, default='gru', required=False)
    parser.add_argument('--seed', help='sys seed', type=int, default=100, required=False)
    parser.add_argument('--n_layers', help='encoder number', type=int, default=1, required=False)
    parser.add_argument('--n_head', help='encoder head number', type=int, default=1, required=False)
    parser.add_argument('--d_k', help='dimention of k', type=int, default=16, required=False)
    parser.add_argument('--d_v', help='dimention of v', type=int, default=16, required=False)
    parser.add_argument('--dropout', help='dropout value', type=float, default=0.1, required=False)
    parser.add_argument('--n_position', help='position encode maximum length', type=int, default=60, required=False)
    parser.add_argument('--d_inner', help='dimention of inner', type=int, default=32, required=False)
    parser.add_argument('--dataset_name', help='choose dataset name', type=str, default="factory_coarse",
                        required=False)

    args = parser.parse_args()
    setup_seed(args.seed)
    print("-----------------args:-----------------")
    print(args)

    # model definition
    if args.model == "rtf_transformer":
        file_predix = "te" + "_" + args.rtf_regressor
    elif args.model == "RNN":
        BI = 'bi' if args.bi else ''
        ATT = 'att' if args.att else ''
        Node = args.node
        file_predix = BI + ATT + Node
    elif args.model == "processtransformer":
        file_predix = "pte"
    elif args.model == "TCN":
        file_predix = "TCN"
    elif args.model == "BiattQRNN":
        file_predix = "BiattQRNN"
    elif args.model == "CQRNN":
        file_predix = "CQRNN"
    else:
        file_predix = ""
        print("wrong model name")
        sys.exit(0)

    # load datasets
    dataset_file_path = dataset_path(dataset_name=args.dataset_name)
    if args.dataset_name == "factory_coarse":
        (X_train, y_train), (X_test, y_test), col_name_index_dict = load_dataset_factory_coarse(dataset_file_path)
        log_file = "../results/CGL-TV/" + file_predix + "log.txt"
        loss_file = "../results/CGL-TV/" + file_predix + "loss.txt"
        pd_file = "../results/CGL-TV/" + file_predix + "_y_test_predict_true.txt"
        max_time_len = 13
        input_feature_dim = 7
        embedding_shape = 26
        best_train_model_name = "../results/CGL-TV/" + file_predix + "best_train_factory_coarse.pkl"
        best_eval_model_name = "../results/CGL-TV/" + file_predix + "best_eval_factory_coarse.pkl"
        min_max = [0, 1, 4]
    elif args.dataset_name == "BPIC-12":
        (X_train, y_train), (X_test, y_test), col_name_index_dict = load_dataset_BPI(dataset_file_path)
        y_train = [y1 / (3600 * 24) for y1 in y_train]
        y_test = [y1 / (3600 * 24) for y1 in y_test]
        log_file = "./bpi/" + file_predix + "log.txt"
        loss_file = "./bpi/" + file_predix + "loss.txt"
        pd_file = "./bpi/" + file_predix + "_y_test_predict_true.txt"
        max_time_len = 73
        input_feature_dim = 8
        embedding_shape = 20
        best_train_model_name = "./bpi/" + file_predix + "best_train_factory_coarse.pkl"
        best_eval_model_name = "./bpi/" + file_predix + "best_eval_factory_coarse.pkl"
        min_max = [0, 1, 2, 5, 6]
    elif args.dataset_name == "helpdesk":
        (X_train, y_train), (X_test, y_test), col_name_index_dict = load_dataset_helpdesk(dataset_file_path)
        y_train = [y1 / (3600 * 24) for y1 in y_train]
        y_test = [y1 / (3600 * 24) for y1 in y_test]
        log_file = "./helpdesk/" + file_predix + "log.txt"
        loss_file = "./helpdesk/" + file_predix + "loss.txt"
        pd_file = "./helpdesk/" + file_predix + "_y_test_predict_true.txt"
        max_time_len = 13
        input_feature_dim = 5
        embedding_shape = 20
        best_train_model_name = "./helpdesk/" + file_predix + "best_train_helpdesk.pkl"
        best_eval_model_name = "./helpdesk/" + file_predix + "best_eval_helpdesk.pkl"
        min_max = [0, 1, 2]
    elif args.dataset_name == "factory_fine":
        (X_train, y_train), (X_test, y_test), col_name_index_dict = load_dataset_factory_fine(dataset_file_path)
        log_file = "../results/FGL-TV/" + file_predix + "_log.txt"
        loss_file = "../results/FGL-TV/" + file_predix + "_loss.txt"
        pd_file = "../results/FGL-TV/" + file_predix + "_y_test_predict_true.txt"
        max_time_len = 31
        input_feature_dim = 7
        embedding_shape = 26
        best_train_model_name = "../results/FGL-TV/" + file_predix + "_best_train_factory_fine.pkl"
        best_eval_model_name = "../results/FGL-TV/" + file_predix + "_best_eval_factory_fine.pkl"
        min_max = [0, 1, 4]
    else:
        print("wrong dataset name")
        sys.exit(0)

    # device and sys output
    device = torch.device(args.device)
    sys.stdout = Logger(log_file)

    # padding as same length
    X_train = padding(X_train, max_time_len, input_feature_dim)
    X_test = padding(X_test, max_time_len, input_feature_dim)

    # min max list of continuous variables
    for index in min_max:
        max_value = np.max(X_train[:, :, index].flatten())
        X_train[:, :, index] = X_train[:, :, index] / max_value
        X_test[:, :, index] = X_test[:, :, index] / max_value

    # split test set and validation set
    split_num = int(X_test.shape[0] / 3)
    X_eval = X_test[:split_num]
    y_eval = y_test[:split_num]
    X_test = X_test[split_num:]
    y_test = y_test[split_num:]

    # Dataset & Dataloader
    dataset_train = TFE_Dataset(X_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataset_test = TFE_Dataset(X_test, y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False)
    dataset_eval = TFE_Dataset(X_eval, y_eval)
    dataloader_eval = DataLoader(dataset_eval, batch_size=128, shuffle=False)

    # model definition
    if args.model == "rtf_transformer":
        model = Transformer_ED(
            n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v, dropout=args.dropout,
            n_position=args.n_position,
            d_inner=args.d_inner, device=device, dataset_name=args.dataset_name,
            col_name_index_dict=col_name_index_dict, squence_max_length=max_time_len, min_max=min_max,
            regressor_type=args.rtf_regressor)
    elif args.model == "TCN":
        # worker + activity with 24
        model = TemporalConvNet(num_inputs=embedding_shape, num_channels=[2, 1], kernel_size=2, dropout=0.2,
                                col_name_index_dict=col_name_index_dict, dataset=args.dataset_name, min_max=min_max)
    elif args.model == "RNN":
        model = RNN(input_size=embedding_shape, hidden_size=10, output_size=1, num_layer=1,
                    col_name_index_dict=col_name_index_dict,
                    BI=args.bi, ATT=args.att, Node=args.node, min_max=min_max, dataset=args.dataset_name)
    elif args.model == "processtransformer":
        model = processtransformer(
            n_layers=1, n_head=4, d_k=64, d_v=64, dropout=0.1,
            d_inner=64, device=device, dataset_name=args.dataset_name,
            col_name_index_dict=col_name_index_dict, squence_max_length=max_time_len, min_max=min_max)
    elif args.model == "BiattQRNN":
        seq_len = max_time_len
        batch_size = 128
        hidden_size = 256
        embedding_dim = embedding_shape
        out_size = 1
        model = qrnn7(hidden_size, out_size, batch_size, embedding_shape, args.dataset_name, col_name_index_dict,
                      min_max)
    elif args.model == "CQRNN":
        batch_size = 128
        seq_len = max_time_len
        hidden_size = 256
        embedding_dim = embedding_shape
        model = BiCRNNAtt(embedding_dim, hidden_size, seq_len, batch_size, args.dataset_name, col_name_index_dict,
                          min_max, 1, 0.3)

    model.to(device)
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, 18, 1000)

    loss_function = torch.nn.L1Loss()

    exc_time = 1000
    # train and test
    if args.mode == "Train":
        early_stopping = EarlyStopping(patience=30, verbose=True, path=best_eval_model_name)
        # save loss in loss_file
        with open(loss_file, "a", encoding='utf-8') as f:
            f.write("epoch" + " " + "loss\n")
        valid_losses = []
        for i in range(args.epochs):
            # train
            model.train()
            start_time = time.time()
            avg_loss = train_epoch(model, dataloader_train, optimizer, loss_function, device)
            end_time = time.time()
            print("training time: ", end_time - start_time)
            if end_time - start_time < exc_time:
                exc_time = end_time - start_time
                print("Min exc time is: ", exc_time)
            print("total_loss of Epoch %s is : %f" % (str(i), avg_loss))
            with open(loss_file, "a", encoding='utf-8') as f:
                f.write(str(i + 1) + " " + str(avg_loss) + "\n")
            valid_losses += [avg_loss]
            # save best model
            if avg_loss <= min(valid_losses):
                torch.save(model, best_train_model_name)
                print('    - [Info] The checkpoint file has been updated.')
            # Test after train
            model.eval()
            valid_loss = eval_epoch(model, dataloader_eval, loss_function, device, pd_file, "train")
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # only test
    elif args.mode == "Test":
        device = torch.device('cuda')
        model = torch.load(best_eval_model_name).to(device)
        model.eval()
        eval_epoch(model, dataloader_test, loss_function, device, pd_file, "test")
    else:
        print("Wrong Mode")
