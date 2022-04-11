import copy
from datetime import datetime
import math
from torch.utils.data import *
from tqdm import tqdm
from torchmetrics import F1Score, Accuracy, Precision, Recall
from Windows_Select import *
from Template import Template
from sklearn.metrics import cohen_kappa_score

# 重写类Dataset，定义数据集
class MyDataset(Dataset):
    # init
    def __init__(self, file_path, device, model, tokenizer, dataset_type, length, random_mask):
        self.type = dataset_type
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.random_mask = random_mask
        self.data = pd.read_csv(file_path, sep="^")
        self.win_split = Windows_Split(length=length)
        self.win_select = Windows_Select(self.model, self.tokenizer, self.device)
        # print(self.data)
        # self.label = os.path.splitext(os.path.basename(file_path))[0]

    def __getitem__(self, index):
        # 模板类
        template = Template(self.device, self.tokenizer)

        # 取数据
        No = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        content = self.data.iloc[index, 2]

        # 切分窗口，返回窗口list
        content_list = self.win_split.split_win(content)

        # 选举窗口
        if self.type == "Train":
            # 获得目标类概率值最大的窗口
            content = self.win_select.select_one(content_list, label, mode='target_max', random_mask=self.random_mask)

            # 套模板，获得token_ids
            token_ids = template.templating(content)
            # 随机mask
            token_ids, targets = template.random_mask(token_ids, label, random=self.random_mask)

            targets = torch.tensor(targets).to(self.device)

            # 序列encode
            token_ids = self.tokenizer.encode_plus(token_ids, padding='max_length', truncation=True,
                                                   return_tensors='pt',
                                                   max_length=512)

            return No, label, token_ids, targets

        # 测试或验证模式，返回所有窗口概率最大的label
        if self.type == 'Test' or self.type == 'Dev':
            pre_label, true_label, loss = self.win_select.select_one(content_list, label, mode='pros_max',
                                                                     random_mask=self.random_mask)
            return pre_label, true_label, loss

    def __len__(self):
        return self.data.shape[0]


class Trainer(object):

    def __init__(self):
        # ------------------------------------------- #
        # google colab parameters
        self.device = torch.device('cuda')
        model_root_dir = r'../Model/chinese_roberta_wwm_ext_pytorch/'
        model_dir = os.path.join(model_root_dir + 'pytorch_model.bin')
        config_dir = os.path.join(model_root_dir + 'bert_config.json')
        vocab_dir = os.path.join(model_root_dir + 'vocab.txt')
        # ------------------------------------------- #

        # # berlin laptop parameters
        #
        # # 定义训练设备
        # self.device = torch.device("cpu")
        # # -----------------需定义-------------------------
        # model_root_dir = r'./Model/chinese_roberta_wwm_ext_pytorch/'
        # model_dir = os.path.join(model_root_dir + 'pytorch_model.bin')
        # config_dir = os.path.join(model_root_dir + 'bert_config.json')
        # vocab_dir = os.path.join(model_root_dir + 'vocab.txt')
        # # ---------------------------------------------

        self.model_config = transformers.BertConfig.from_pretrained(pretrained_model_name_or_path=config_dir)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_name_or_path=vocab_dir)
        self.model = transformers.BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_dir,
                                                                  config=self.model_config)

        # 将模型放到指定设备上
        self.model.to(self.device)
        # print(self.model)

        # # 冻结参数
        # print(self.model.bert.encoder)
        # for param in self.model.bert.encoder.parameters():
        #     param.requires_grad = False

        # -----------------需定义-------------------------
        # 定义训练参数
        self.numclass = 14
        self.max_len = 512
        self.window_length = 490
        self.batch_size = 16
        self.epoch = 100
        self.learning_rate = 5e-6
        self.weight_decay = 0.0005
        self.decay_rate = 0.98
        self.early_stop = 5
        self.random_mask = False
        print(self.learning_rate)

        self.data_dir = '../THUCNews/data_2.4%/'
        self.model_save_dir = './Model_template_11_data_2.4%/'
        # ---------------------------------------------

        # 加载数据集
        self.train_dataset = MyDataset(os.path.join(self.data_dir + 'train.csv'), self.device, self.model,
                                       self.tokenizer, 'Train', self.window_length, self.random_mask)
        self.dev_dataset = MyDataset(os.path.join(self.data_dir + 'dev.csv'), self.device, self.model, self.tokenizer,
                                     'Dev', self.window_length, self.random_mask)
        self.test_dataset = MyDataset(os.path.join(self.data_dir + 'test.csv'), self.device, self.model, self.tokenizer,
                                      'Test', self.window_length, self.random_mask)

        # 实例化数据加载器
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       drop_last=True)
        self.dev_loader = DataLoader(dataset=self.dev_dataset, batch_size=self.batch_size * 10, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size * 10, shuffle=True)

    # 保存checkpoint方法
    def save_checkpoint(self, epoch_index, dev_ave_f1, test_ave_f1, dev_acc, test_acc, dev_ave_pre, test_ave_pre,
                        dev_ave_re, test_ave_re, dev_kappa, test_kappa, dev_cla_f1, test_cla_f1,
                        dev_cla_pre, test_cla_pre, dev_cla_re, test_cla_re):

        ckpt_name = "epoch_{}_test_f1_{}_test_acc_{}_test_kappa_{}.ckpt". \
            format(epoch_index, test_ave_f1, test_acc, test_kappa)

        os.makedirs(self.model_save_dir, exist_ok=True)
        path = os.path.join(self.model_save_dir + ckpt_name)

        checkpoint = {'epoch': epoch_index,
                      'model_state_dict': self.model.state_dict(),
                      'dev_macro_F1': dev_ave_f1,
                      'test_macro_F1': test_ave_f1,
                      'dev_accuracy': dev_acc,
                      'test_accuracy': test_acc,
                      'dev_kappa': dev_kappa,
                      'test_kappa': test_kappa,
                      'dev_macro_precision': dev_ave_pre,
                      'test_macro_precision': test_ave_pre,
                      'dev_macro_recall': dev_ave_re,
                      'test_macro_recall': test_ave_re,
                      'dev_class_F1': dev_cla_f1,
                      'test_class_F1': test_cla_f1,
                      'dev_class_precision': dev_cla_pre,
                      'test_class_precision': test_cla_pre,
                      'dev_class_recall': dev_cla_re,
                      'test_class,recall': test_cla_re,
                      'ckpt_name': ckpt_name,
                      'time': datetime.now()}

        torch.save(checkpoint, path)
        print("#-----------Checkpoint {} saved.-----------#".format(ckpt_name))
        print('')

    # 计算指标
    def compute_metric(self, pred, true):

        aver_f1 = F1Score(num_classes=self.numclass, average='macro').to(self.device)
        aver_precision = Precision(num_classes=self.numclass, average='macro').to(self.device)
        aver_recall = Recall(num_classes=self.numclass, average='macro').to(self.device)
        accuracy = Accuracy(num_classes=self.numclass, average='weighted').to(self.device)

        cla_f1 = F1Score(num_classes=self.numclass, average=None).to(self.device)
        cla_precision = Precision(num_classes=self.numclass, average=None).to(self.device)
        cla_recall = Recall(num_classes=self.numclass, average=None).to(self.device)

        average_f1 = aver_f1(pred, true)
        average_precision = aver_precision(pred, true)
        average_recall = aver_recall(pred, true)
        acc = accuracy(pred, true)

        class_f1 = cla_f1(pred, true)
        class_precision = cla_precision(pred, true)
        class_recall = cla_recall(pred, true)

        pred_cpu = copy.deepcopy(pred).to(torch.device('cpu'))
        true_cpu = copy.deepcopy(true).to(torch.device('cpu'))
        kappa = cohen_kappa_score(pred_cpu, true_cpu)

        return acc, kappa, average_f1, average_precision, average_recall, class_f1, class_precision, class_recall

    # 评估模型
    def evaluate(self, epoch_index, eva_type, full_type):

        self.model.eval()

        if eva_type == 'Dev':
            loader = self.dev_loader

        if eva_type == 'Test':
            loader = self.test_loader

        global_loss = 0.0
        pre_labels = []
        true_labels = []
        # TP = [0] * len(template.categories)
        # FP = [0] * len(template.categories)
        # FN = [0] * len(template.categories)
        # F1 = [0.0] * len(template.categories)
        # precision = [0.0] * len(template.categories)
        # recall = [0.0] * len(template.categories)
        # F1_sum, pre_sum, re_sum, TP_sum = 0, 0, 0, 0
        total_predict = 0

        # 不计算梯度
        with torch.no_grad():
            # 进度条
            pbar_eva = tqdm(total=len(loader))

            for data in loader:
                pre_label, true_label, loss = data
                loss = loss.tolist()
                pre_label = pre_label.tolist()
                # true_label = true_label.tolist()
                for l in loss:
                    global_loss += l
                for p in pre_label:
                    pre_labels.append(p)
                for t in true_label:
                    true_labels.append(int(t))
                pbar_eva.update(1)
            pbar_eva.close()
            # print('')

        # 转化为张量
        pre_label_tensor = torch.IntTensor(pre_labels).to(self.device)
        true_label_tensor = torch.IntTensor(true_labels).to(self.device)

        acc, kappa, average_f1, average_precision, average_recall, class_f1, class_precision, class_recall = \
            self.compute_metric(pre_label_tensor, true_label_tensor)

        # Macro_F1 = F1_sum / total_predict
        # Macro_Precision = pre_sum / total_predict
        # Macro_Recall = re_sum / total_predict
        # acc = TP_sum / total_predict

        # print('')
        print("#----Epoch {} Eva_type: {} Loss: {} Macro_F1: {} Accurancy: {} Kappa: {} ----#".
              format(epoch_index, eva_type, global_loss, average_f1, acc, kappa))

        if full_type:
            return global_loss, average_f1, average_precision, average_recall, acc, kappa, class_f1, class_precision, class_recall
        else:
            return global_loss, average_f1, acc, kappa

    def training(self):
        early_stop_current = 0
        best_dev_ave_F1 = 0

        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.decay_rate)

        for epoch_index in range(self.epoch):
            if epoch_index == 0:
                print('#--------------------Evaluate Dev--------------------#')
                dev_loss, dev_ave_f1, dev_ave_pre, dev_ave_re, dev_acc, dev_kappa, dev_cla_f1, dev_cla_pre, dev_cla_re \
                    = self.evaluate(epoch_index, 'Dev', full_type=True)
                best_dev_ave_F1 = dev_ave_f1
                print('#--------------------Evaluate Test--------------------#')
                test_loss, test_ave_f1, test_ave_pre, test_ave_re, test_acc, test_kappa, test_cla_f1, test_cla_pre, \
                test_cla_re = self.evaluate(epoch_index, 'Test', full_type=True)

            print("#--------------------第{}轮训练--------------------#".format(epoch_index + 1))

            pbar_train = tqdm(total=math.floor(len(self.train_loader)))

            total_loss = 0

            self.model.train()

            for data in self.train_loader:
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                ids, label, tokens_tensor, targets = data
                ids = ids.to(self.device)
                tokens_tensor = transformers.BatchEncoding(tokens_tensor)
                tokens_tensor.to(self.device)
                # 降维
                tokens_tensor['input_ids'] = tokens_tensor['input_ids'].squeeze(1)
                tokens_tensor['token_type_ids'] = tokens_tensor['token_type_ids'].squeeze(1)
                tokens_tensor['attention_mask'] = tokens_tensor['attention_mask'].squeeze(1)
                targets.to(self.device)
                # 输入模型
                output = self.model(**tokens_tensor, labels=targets)
                loss = output.loss
                logits = output.logits
                total_loss += loss.item()

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()

                pbar_train.update(1)

            pbar_train.close()
            # print('')

            lr_scheduler.step()

            # early_stop
            print('#--------------------Evaluate Dev--------------------#')
            dev_loss, dev_ave_f1, dev_ave_pre, dev_ave_re, dev_acc, dev_kappa, dev_cla_f1, dev_cla_pre, dev_cla_re \
                = self.evaluate(epoch_index, 'Dev', full_type=True)

            if epoch_index > 0 and (dev_ave_f1 > best_dev_ave_F1):
                print('#--------------------Evaluate Test--------------------#')
                test_loss, test_ave_f1, test_ave_pre, test_ave_re, test_acc, test_kappa, test_cla_f1, test_cla_pre, \
                test_cla_re = self.evaluate(epoch_index, 'Test', full_type=True)

                self.save_checkpoint(epoch_index, dev_ave_f1, test_ave_f1, dev_acc, test_acc, dev_ave_pre, test_ave_pre,
                                     dev_ave_re, test_ave_re, dev_kappa, test_kappa, dev_cla_f1, test_cla_f1,
                                     dev_cla_pre, test_cla_pre, dev_cla_re, test_cla_re)
                early_stop_current = 0
                best_dev_ave_F1 = dev_ave_f1

            else:
                early_stop_current += 1
                # 若连续不能提升模型效果，打断训练
                if early_stop_current >= self.early_stop:
                    self.save_checkpoint(epoch_index, dev_ave_f1, test_ave_f1, dev_acc, test_acc, dev_ave_pre,
                                         test_ave_pre, dev_ave_re, test_ave_re, dev_kappa, test_kappa, dev_cla_f1,
                                         test_cla_f1, dev_cla_pre, test_cla_pre, dev_cla_re, test_cla_re)
                    print("#------Early stopping at epoch {}.--------#".format(epoch_index))
                    return None

        self.save_checkpoint(epoch_index, dev_ave_f1, test_ave_f1, dev_acc, test_acc, dev_ave_pre, test_ave_pre,
                             dev_ave_re, test_ave_re, dev_kappa, test_kappa, dev_cla_f1, test_cla_f1,
                             dev_cla_pre, test_cla_pre, dev_cla_re, test_cla_re)
        return None


def main():
    trainer = Trainer()
    trainer.training()


if __name__ == '__main__':
    main()
