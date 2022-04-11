import numpy as np


class Template(object):

    def __init__(self, device, tokenizer):
        super(Template, self).__init__()

        self.device = device
        self.tokenizer = tokenizer

        # 定义模板
        # 若修改模板长度，需修改三个位置
        self.mask_index = 5
        self.template = ['[unused%s]' % i for i in range(1, 11)]
        self.template.insert(self.mask_index, '[MASK]')
        # self.template.append('。')
        self.template_ids = [tokenizer.convert_tokens_to_ids(token) for token in self.template]

        self.j = 11
        # 定义映射
        self.caijing = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j))
        self.caipiao = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+1))
        self.fangchan = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+2))
        self.jiaju = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+3))
        self.jiaoyu = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+4))
        self.keji = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+5))
        self.shehui = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+6))
        self.shishang = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+7))
        self.shizheng = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+8))
        self.tiyu = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+9))
        self.xingzuo = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+10))
        self.youxi = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+11))
        self.yule = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+12))
        self.gupiao = tokenizer.convert_tokens_to_ids('[unused%s]' % str(self.j+13))

        self.categories = {
            '财经': 'caijing',
            '彩票': 'caipiao',
            '房产': 'fangchan',
            '家居': 'jiaju',
            '教育': 'jiaoyu',
            '科技': 'keji',
            '社会': 'shehui',
            '时尚': 'shishang',
            '时政': 'shizheng',
            '体育': 'tiyu',
            '星座': 'xingzuo',
            '游戏': 'youxi',
            '娱乐': 'yule',
            '股票': 'gupiao'
        }

        self.categories_id = {
            '财经': '0',
            '彩票': '1',
            '房产': '2',
            '家居': '3',
            '教育': '4',
            '科技': '5',
            '社会': '6',
            '时尚': '7',
            '时政': '8',
            '体育': '9',
            '星座': '10',
            '游戏': '11',
            '娱乐': '12',
            '股票': '13'
        }

    # 为数据添加模板
    def templating(self, seq):
        token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
        token_ids = self.template_ids + token_ids

        return token_ids

    # 随机替换掉一些token
    def random_mask(self, token_ids, label, random):
        # 输出的token id序列
        output = []
        # 输出的target（label）序列
        # 添加[CLS]赋值
        labels = [-100]
        # 若随机mask
        if random == True:
            # 随机序列
            random_segment = np.random.random(len(token_ids))
            # 将随机数和token id捆绑在一起，进行随机替换
            '''
            !!!!!!该部分需要实验!!!!!!!!!!
            1、no random mask
            2、random mask，不填充labels
            3、random mask，填充labels，让mask部分参与loss计算
            '''
            for ran, token_id in zip(random_segment, token_ids):
                # 15% 中的 80% 被替换为 【MASK】
                if ran < 0.15 * 0.8:
                    output.append((self.tokenizer.mask_token_id))
                    labels.append(token_id)
                # 15% 中的 10% 被替换成随机词
                elif ran < 0.15 * 0.9 and ran >= 0.15 * 0.8:
                    output.append(np.random.randint(low=106, high=self.tokenizer.vocab_size))
                    labels.append(token_id)
                # 15% 中的 10% 原封不动
                elif ran < 0.15 and ran >= 0.15 * 0.9:
                    output.append(token_id)
                    labels.append(token_id)
                # 剩下的不替换
                else:
                    output.append(token_id)
                    labels.append(-100)
        else:
            output = token_ids

        # 若长度超过512，则截断
        if len(labels) >= 512:
            labels = labels[0:510]

            # 补充labels长度至512
        labels += [-100] * (512 - len(labels))

        # 保证labels末尾为-100
        labels[511] = -100

        # 更新target
        target = int(self.categories_id[label]) + self.j
        # 保证MASK位置id为103
        output[self.mask_index] = 103
        # 序列encode后需插入[CLS],故往后移一位
        labels[self.mask_index + 1] = target

        return output, labels
