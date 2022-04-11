import re
import pandas as pd


filename = ("财经", "彩票", "房产", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐", "股票")
start_index = (798977, 256822, 264410, 224236, 284460, 481650, 430801, 326396, 339764, 0, 402850, 406428, 131604, 644579)
# # end_index = (798999, 256833, 264510, 256821, 326395, 644578, 481649, 339763, 402849, 131603, 406427, 430800, 224235, 798976)
end_index = (836074, 264409, 284459, 256821, 326395, 644578, 481649, 339763, 402849, 131603, 406427, 430800, 224235, 798976)

class Windows_Split(object):

    def __init__(self, length):
        super(Windows_Split, self).__init__()

        self.sent_length = length

    def split_sent(self, txt):
        txt = re.sub('^.(\s+)([^“‘\"\'])', r"\2", txt)  # 空格前无字符，将空格去掉
        # txt = re.sub('([^\s，。！？；、：’“])\s+([^“‘\"\'])', r"\1,\2", txt)  # 空格前有字符，将空格替换成"，"
        txt = re.sub('([。！!？；\?])([^“‘\"\'])', r"\1\n\2", txt)  # 遇到句号、感叹号、问号，进行切割
        txt = re.sub('(-{2})([^“‘\"\'])', r"\1\n\2", txt)  # 遇到--，进行切割
        txt = re.sub('(\.{6})([^“‘\"\'])', r"\1\n\2", txt)  # 遇到......，进行切割
        txt = re.sub('(…)([^“‘\"\'…])', r"\1\n\2", txt)  # 遇到……，进行切割
        txt = re.sub('([。！!？\?][\"\'“‘])([^，,。！!？\?])', r'\1\n\2', txt)  # 双引号前有终止符，双引号才是句子终点
        txt = txt.rstrip()
        txt = txt.split("\n")
        txt = list(filter(None, txt))
        return txt

    # # 切割句子测试
    # x = input()
    # x = split_sent(x)
    # print(x)

    def split_win(self, content):

        temp = self.split_sent(content)
        text = ''
        text_list = []

        # 对已切分的句子list进行遍历
        for t in temp:
            if len(text) + len(t) <= self.sent_length:
                text += t
            else:
                # 若上一个text已满，则将text放进输出list里，再清空text，存放句子
                text_list.append(text)
                text = ''
                text += t
        # 最后输出text
        text_list.append(text)

        return text_list


def main():
    windows_split = Windows_Split(length=480)

    test = pd.read_csv('./test.csv', sep='^')

    print(test)

    print(test.iloc[0, 2])
    print(type(test.iloc[0, 2]))

    text_list = windows_split.win_split(test.iloc[2, 2])

    print(text_list)

    exit(0)


if __name__ == '__main__':
    main()
