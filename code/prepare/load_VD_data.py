import json
import os.path as osp
import nltk
import copy


BOD = '<BOD>'       # beginning of dialog
EOS = '<EOS>'       # end of sentence
BOS = '<BOS>'       # beginning of sentence
UNK = '<UNK>'       # unknown word
PAD = '<PAD>'       # padding
LOC = '<LOC>'       # located the object
NOC = '<NOC>'       # cannot located the object

def simple_tokenize(sentences):
    '''
    tokenize sentences
    :param sentences:[s1, s2, ...]
    :return:[[s1 tokens], [s2 tokens]]
    '''
    processed = []
    for s in sentences:
        txt = nltk.tokenize.word_tokenize(s.lower())
        processed.append(txt)
    return processed

def check_sentLength(sents):
    '''
    statistic sentence length
    :param sents: [[s1 tokens], [s2 tokens]]
    '''
    sent_lengths = {}
    for tokens in sents:
        nw = len(tokens)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length of sentence in raw data is %d' % max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    acc = 0  # accumulative distribution
    for i in range(max_len+1):
        acc += sent_lengths.get(i, 0)
        print('%2d: %10d %.3f%% %.3f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/sum_len, acc*100.0/sum_len))
    print('\n')

def build_vocab(data_root, data_type, min_word_count=5, num_words=2000, all=True):
    '''
    build vocab list from data file
    :param data_file:
    :param min_word_count:
    :param num_words:
    :return:
    '''
    counts = {}
    dial_list = []
    dial_list = const_dialog(get_data_file(data_root, data_type, 'train'), dial_list)   # only training set
    if all:
        dial_list = const_dialog(get_data_file(data_root, data_type, 'val'), dial_list)
        if 'unc' in data_root:
            dial_list = const_dialog(get_data_file(data_root, data_type, 'testA'), dial_list)
            dial_list = const_dialog(get_data_file(data_root, data_type, 'testB'), dial_list)

    print('the sum of dialog sent_num:', len(dial_list))
    check_sentLength(dial_list)

    for sent in dial_list:
        for w in sent:
            counts[w] = counts.get(w, 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    if min_word_count > 1:
        vocab = [w for (c, w) in cw if c >= min_word_count]
    elif num_words > 0:
        vocab = [w for (_, w) in cw[:num_words]]

    vocab = [PAD, BOD, BOS] + vocab + [UNK, LOC, NOC, EOS]                # add the padding and eos and unknow token

    print('vocab length:', len(vocab), '\n')

    return vocab

def const_dialog(data_file, dial_list):
    data = json.load(open(data_file, 'r'))
    for dial_id, dial in data['dials'].items():
        dial_token = dial['dial_token']
        for tokens in dial_token:
            dial_list.append(tokens)
    return dial_list

def get_data_file(data_root, type, split):
    '''

    :param data_root:
    :param type:
    :param split:
    :return:
    '''
    if type == 'combine':
        data_file = osp.join(data_root, 'dump_c_' + split + '.json')
    elif type == 'single':
        data_file = osp.join(data_root, 'dump_s_' + split + '.json')
    elif type == 'multi':
        data_file = osp.join(data_root, 'dump_m_' + split + '.json')

    return data_file

def load_json(file):
    data = json.load(open(file, 'r'))
    return data

class JsonLoader_append(object):
    def __init__(self, path, split):
        self.path = path
        self.m_vd_file = osp.join(self.path, 'm_'+ split + '.json')
        self.s_vd_file = osp.join(self.path, 's_' + split + '.json')
        self.ann_file = osp.join(self.path,  '../instances.json')
        self.split = split
        # info all type data same
        self.anns_dict = {}

        # info single and combine data same
        self.sc_imgs_dict = {}
        self.m_imgs_dict = {}

        self.sc_imgs2refs_dict = {}
        self.m_imgs2refs_dict = {}

        self.sc_imgs2cands_dict = {}
        self.m_imgs2cands_dict = {}

        self.sc_ref_ids_list = []
        self.m_ref_ids_list = []

        # info three types are different
        self.s_refs_dict = {}
        self.m_refs_dict = {}
        self.c_refs_dict = {}

        self.s_dials_dict = {}
        self.m_dials_dict = {}
        self.c_dials_dict = {}

        self.s_dial_ids_list = []
        self.m_dial_ids_list = []
        self.c_dial_ids_list = []

    def process(self):
        # get anns from ann_file
        ann_data = load_json(self.ann_file)
        ann_item = {}
        for ann_info in ann_data['annotations']:
            ann_item['bbox'] = ann_info['bbox']
            ann_item['category_id'] = ann_info['category_id']
            self.anns_dict[ann_info['id']] = copy.copy(ann_item)
        self.categories = ann_data['categories']

        # get data from single round dialog file
        s_data = load_json(self.s_vd_file)['data']
        s_dial_data = s_data['dialogs']
        s_speakers = s_data['speakers']
        s_listeners = s_data['listeners']
        s_speaker_tokens = simple_tokenize(s_speakers)
        s_listener_tokens = simple_tokenize(s_listeners)
        print('The total number of single-round dialogs:', len(s_dial_data))

        # info same in three types
        self.sents = s_data['REs']
        self.entities = s_data['Entities']

        # get data from multi round dialog file
        m_data = load_json(self.m_vd_file)['data']
        m_dial_data = m_data['dialogs']
        m_speakers = m_data['speakers']
        m_listeners = m_data['listeners']
        m_speaker_tokens = simple_tokenize(m_speakers)
        m_listener_tokens = simple_tokenize(m_listeners)
        print('The total number of multiple-round dialogs:', len(m_dial_data))

        # get candidates info from RE data (single round data file)
        for item in s_dial_data:
            img_id = item['image_id']
            self.sc_imgs2cands_dict[img_id] = copy.copy(item['candidate'])
        #print('all image number', len(self.sc_imgs2cands_dict))
        # get candidates info from multi round data file
        for item in m_dial_data:
            img_id = item['image_id']
            self.m_imgs2cands_dict[img_id] = copy.copy(item['candidate'])
        #print('image number in multi', len(self.m_imgs2cands_dict))

        # get image info from instance.json and single round file
        img_item = {}
        for img in ann_data['images']:
            # instance.json contain all image in data, while single round file splits into train, val, testA, testB
            img_id = img['id']
            if img['id'] in self.sc_imgs2cands_dict.keys():
                img_item['h'] = img['height']
                img_item['w'] = img['width']
                img_item['file'] = img['file_name']
                img_item['cand_ids'] = self.sc_imgs2cands_dict[img_id]
                self.sc_imgs_dict[img_id] = copy.deepcopy(img_item)
            if img['id'] in self.m_imgs2cands_dict.keys():
                img_item['h'] = img['height']
                img_item['w'] = img['width']
                img_item['file'] = img['file_name']
                img_item['cand_ids'] = self.m_imgs2cands_dict[img_id]
                self.m_imgs_dict[img_id] = copy.deepcopy(img_item)
        print('all image number', len(self.sc_imgs_dict))
        print('image number in multi', len(self.m_imgs_dict))

        ref_item = {}
        dial_item = {}
        round_count = {}
        re_list_s = []
        

        # construct ref_info dialog_info img2refs info for single round dialog
        for item in s_dial_data:
            # construct ref_info
            ref_id = item['ann_id']                 # only anns with RE, it will have dialog
            img_id = item['image_id']
            dial_id = item['dialog_id']
            # single round ref_dict
            if ref_id in self.s_refs_dict.keys():
                self.s_refs_dict[ref_id]['dial_ids'].append(dial_id)
                if item['re_id'] in self.s_refs_dict[ref_id]['re_ids']:
                    pass
                else:
                    self.s_refs_dict[ref_id]['re_ids'].append(item['re_id'])
                    re_list_s.append(item['re_id'])
            else:
                ref_item['img_id'] = img_id
                ref_item['re_ids'] = [item['re_id']]
                re_list_s.append(item['re_id'])
                ref_item['dial_ids'] = [dial_id]
                self.s_refs_dict[ref_id] = copy.deepcopy(ref_item)
                # img2refs dict
                if img_id in self.sc_imgs2refs_dict.keys():
                    self.sc_imgs2refs_dict[img_id].append(ref_id)
                else:
                    self.sc_imgs2refs_dict[img_id] = [ref_id]
            # dialog dict
            dial_item['rounds'] = item['rounds']
            dial_item['re_id'] = item['re_id']
            dial_item['ann_id'] = item['ann_id']
            round_count[item['rounds']] = round_count.get(item['rounds'], 0) + 1
            dial = []
            dial_token = []
            for per_round in item['dialog']:
                s = s_speakers[per_round['speaker']]
                l = s_listeners[per_round['listener']]
                s_token = s_speaker_tokens[per_round['speaker']]
                l_token = s_listener_tokens[per_round['listener']]
                dial.append(s)
                dial.append(l)
                dial_token.append(s_token)
                dial_token.append(l_token)
            dial_item['dial'] = dial
            dial_item['dial_token'] = dial_token

            self.s_dials_dict[dial_id] = copy.deepcopy(dial_item)
            self.s_dial_ids_list.append(dial_id)
        #print('refs_dict', self.s_refs_dict[1555237])
        # combine = single + multi
        self.c_refs_dict = copy.deepcopy(self.s_refs_dict)
        self.c_dials_dict = copy.deepcopy(self.s_dials_dict)
        self.c_dial_ids_list =copy.deepcopy( self.s_dial_ids_list)

        # construct ref_info dialog_info img2refs info for multi round dialog
        for item in m_dial_data:
            # construct ref_info
            ref_id = item['ann_id']                 # only anns with RE, it will have dialog
            img_id = item['image_id']
            dial_id = item['dialog_id']
            if ref_id in self.m_refs_dict.keys():
                self.m_refs_dict[ref_id]['dial_ids'].append(dial_id)
                self.c_refs_dict[ref_id]['dial_ids'].append(dial_id)
                if item['re_id'] in self.m_refs_dict[ref_id]['re_ids']:
                    pass
                else:
                    self.m_refs_dict[ref_id]['re_ids'].append(item['re_id'])
            else:
                ref_item['img_id'] = img_id
                ref_item['re_ids'] = self.c_refs_dict[ref_id]['re_ids'] # multi data lack of REs that only single round
                ref_item['dial_ids'] = [dial_id]
                self.m_refs_dict[ref_id] = copy.deepcopy(ref_item)
                self.c_refs_dict[ref_id]['dial_ids'].append(dial_id)    # add multi-round dialog
                # img2refs dict
                if img_id in self.m_imgs2refs_dict.keys():
                    self.m_imgs2refs_dict[img_id].append(ref_id)
                else:
                    self.m_imgs2refs_dict[img_id] = [ref_id]
            # dialog dict
            dial_item['rounds'] = item['rounds']
            dial_item['re_id'] = item['re_id']
            dial_item['ann_id'] = item['ann_id']
            round_count[item['rounds']] = round_count.get(item['rounds'], 0) + 1
            dial = []
            dial_token = []
            for per_round in item['dialog']:
                s = m_speakers[per_round['speaker']]
                l = m_listeners[per_round['listener']]
                s_token = m_speaker_tokens[per_round['speaker']]
                l_token = m_listener_tokens[per_round['listener']]
                dial.append(s)
                dial.append(l)
                dial_token.append(s_token)
                dial_token.append(l_token)
            dial_item['dial'] = dial
            dial_item['dial_token'] = dial_token
            self.m_dials_dict[dial_id] = copy.deepcopy(dial_item)
            self.m_dial_ids_list.append(dial_id)
            self.c_dials_dict[dial_id] = copy.deepcopy(dial_item)
            self.c_dial_ids_list.append(dial_id)

        # print distribution of all dialogs
        max_len = max(round_count.keys())
        print('max round num of dialog in raw data is %d' % max_len)
        print('dialog round distribution (count, number of words):')
        sum_len = sum(round_count.values())
        acc = 0  # accumulative distribution
        for i in range(1, max_len + 1):
            acc += round_count.get(i, 0)
            print('%2d: %10d %.3f%% %.3f%%' % (
            i, round_count.get(i, 0), round_count.get(i, 0) * 100.0 / sum_len, acc * 100.0 / sum_len))

        print('all refer number', len(self.s_refs_dict), len(self.c_refs_dict))
        print('refer number in multi', len(self.m_refs_dict))
        print('single dialog number', len(self.s_dials_dict), len(self.s_dial_ids_list))
        print('multi dialog number', len(self.m_dials_dict), len(self.m_dial_ids_list))
        print('combine dialog number', len(self.c_dials_dict), len(self.c_dial_ids_list))

        for ref_id in self.s_refs_dict.keys():
            self.sc_ref_ids_list.append(ref_id)
        for ref_id in self.m_refs_dict.keys():
            self.m_ref_ids_list.append(ref_id)
        print('#(target-RE pair) in single', len(re_list_s))
        print('load finish')

        c_dict = {
            'imgs': self.sc_imgs_dict,
            'anns': self.anns_dict,
            'cates': self.categories,
            'refs': self.c_refs_dict,
            'dials': self.c_dials_dict,
            'imgs2refs': self.sc_imgs2refs_dict,
            'imgs2cands': self.sc_imgs2cands_dict,
            'ref_ids': self.sc_ref_ids_list,
            'dial_ids': self.c_dial_ids_list,
            'sents': self.sents,
            'entities': self.entities
        }
        json.dump(c_dict, open(osp.join(self.path, 'dump_c_' + self.split + '.json'), 'w'))
        print('dump_c_', self.split, '.json dump finish')

if __name__ == '__main__':

    Path = '../../data/refcoco/unc_1222_split'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../../data/refcoco/unc_split')
    print('----------------------------------Train split----------------------------------')
    jsonloader = JsonLoader_append(Path,  'train')
    jsonloader.process()
    print('----------------------------------Val split----------------------------------')
    jsonloader = JsonLoader_append(Path,  'val')
    jsonloader.process()
    print('----------------------------------TestA split----------------------------------')
    jsonloader = JsonLoader_append(Path,  'testA')
    jsonloader.process()
    print('----------------------------------TestB split----------------------------------')
    jsonloader = JsonLoader_append(Path,  'testB')
    jsonloader.process()

    print('----------------------------------build vocab----------------------------------')
    print('combine vocab')
    vocab = build_vocab(Path, 'combine', 5, 2000, True)
    json.dump({'vocab': vocab}, open(osp.join(Path, 'c_vocab.json'), 'w'))
