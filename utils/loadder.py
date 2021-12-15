def get_wav_list(filename):
    txt_obj = open(filename, 'r')  # 打开文件并读入
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')  # 文本分割
    dic_wavlist = {}  # 初始化字典
    dic_imglist = {}  # 初始化字典
    list_wavmark = []  # 初始化wav列表
    for i in txt_lines:
        if (i != ''):
            txt_l = i.split(' ')
            dic_wavlist[txt_l[0]] = txt_l[2]
            dic_imglist[txt_l[0]] = txt_l[1]
            list_wavmark.append(txt_l[0])
    txt_obj.close()
    return dic_wavlist,dic_imglist,list_wavmark
# y = get_wav_list('../data/train.txt')
