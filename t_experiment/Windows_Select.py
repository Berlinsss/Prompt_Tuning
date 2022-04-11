import os

import pandas as pd
import torch
import transformers
from Windows_Split import Windows_Split

from Template import *


class Windows_Select(object):

    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # 选一个新闻窗口代表
    def select_one(self, content_list, label, mode, random_mask):

        # 定义tokens_tensor形状，并初始化
        shape = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        tokens_tensor = transformers.BatchEncoding(shape, tensor_type='pt').to(self.device)
        targets_tensor = torch.LongTensor().to(self.device)

        template = Template(self.device, self.tokenizer)

        for content in content_list:
            # 套模板
            token_ids = template.templating(content)

            # 随机mask
            token_ids, target_ids = template.random_mask(token_ids, label, random=random_mask)

            target_tensor = torch.LongTensor([target_ids]).to(self.device)

            targets_tensor = torch.cat((targets_tensor, target_tensor), 0)

            # 编码
            token_tensor = self.tokenizer.encode_plus(token_ids, padding='max_length', truncation=True,
                                                      return_tensors='pt',
                                                      max_length=512).to(self.device)

            # 添加至BatchEncoding
            tokens_tensor['input_ids'] = torch.cat((tokens_tensor['input_ids'].long(), token_tensor['input_ids']),
                                                   dim=0)
            tokens_tensor['token_type_ids'] = torch.cat((tokens_tensor['token_type_ids'].long(), token_tensor['token_type_ids']), dim=0)
            tokens_tensor['attention_mask'] = torch.cat((tokens_tensor['attention_mask'].long(), token_tensor['attention_mask']), dim=0)


        self.model.eval()

        # 不计算梯度
        with torch.no_grad():
            predictions = self.model(**tokens_tensor, labels=targets_tensor)

        # 预测分数
        predicted_logits = predictions.logits

        # 获取label的index
        label_index = template.categories_id[label]
        label_index = int(label_index)

        # 只取[MASK]部分
        logits = predicted_logits[:, template.mask_index + 1, template.caijing:template.gupiao + 1 ]

        # 清空缓存
        torch.cuda.empty_cache()

        # 各窗口概率最大模式，返回整条新闻的预测label下标, 真实label下标, loss
        if mode == 'pros_max':
            # 求最大值下标
            max_index = logits.argmax()
            max_index = int(max_index)

            # 取最可能的label
            most_poss_label = max_index % 14

            return most_poss_label, label_index, predictions.loss

        if mode == 'target_max':
            # 获取 该新闻正确标签的概率
            pros = logits[:, label_index]

            # 目标类概率最大的内容下标
            max_index = pros.argmax()
            max_index = int(max_index)

            return content_list[max_index]


def main():
    device = torch.device("cuda")

    model_root_dir = r'./Model/chinese_roberta_wwm_ext_pytorch/'
    model_dir = os.path.join(model_root_dir + 'pytorch_model.bin')
    config_dir = os.path.join(model_root_dir + 'bert_config.json')
    vocab_dir = os.path.join(model_root_dir + 'vocab.txt')

    model_config = transformers.BertConfig.from_pretrained(pretrained_model_name_or_path=config_dir)
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_name_or_path=vocab_dir)
    model = transformers.BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_dir,
                                                         config=model_config)
    # 将模型放到指定设备上
    model.to(device)

    data_import_dir = './test.csv'

    windows_select = Windows_Select(model, tokenizer, device)

    windows_split = Windows_Split(length=480)

    df = pd.read_csv(data_import_dir, sep="^")

    for index, row in df.iterrows():
        content_list = windows_split.split_win(row['content'])
        out = windows_select.select_one(content_list, row['label'], mode='target_max')
        print(out)


if __name__ == '__main__':
    main()
