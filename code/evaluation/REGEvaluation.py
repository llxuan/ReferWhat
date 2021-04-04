from .tokenizer import PTBTokenizer
from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider

remove_list = ['there is ','it is','here is','there is a','it is a','here is a','there is an','it is an','here is an',
               'please locate ', 'in the picture', 'the color of ', 'the location of', 'the size is']

all_possible_arrange ={3:[[0,1,2],[0,2,1],[1,2,0],[1,0,2],[2,0,1],[2,1,0]],
                       2:[[0,1],[1,0]],
                       1:[[0]]}

class REGEvaluation(object):
    def __init__(self, refer, pred, dialog):
        '''
        :param refer: gt re {id: re}
        :param pred: dialog speaker part
        :param dialog: generate dialog {id, dialog}
        '''

        self.evalRefs = []      # only eval list
        self.refToEval = {}     # [ref_id, eval results]
        self.refer = refer
        self.pred = pred
        self.dialog = dialog

    def re_eval(self, pred, gt):
        '''
        :param pred: Speakers prediction concate ['s1 s2 s3']
        :param gt:[RE]
        '''
        #print('tokenization...')
        tokenizer = PTBTokenizer()
        self.pred_tokens = tokenizer.tokenize(pred)
        self.gt_tokens = tokenizer.tokenize(gt)
        # =================================================
        # Set up scorers
        # =================================================
        #print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        # =================================================
        # Compute scores
        # =================================================
        eval_result = {}
        for scorer, method in scorers:
            # print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.gt_tokens, self.pred_tokens)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    eval_result[m] = sc
                    self.setRefToEvalRefs(scs, self.gt_tokens.keys(), m)
                    # print ("%s: %0.3f"%(m, sc))

            else:
                eval_result[method] = score
                self.setRefToEvalRefs(scores, self.gt_tokens.keys(), method)
                # print ("%s: %0.3f"%(method, score))

        return eval_result

    def setRefToEvalRefs(self, scores, refIds, method):
        for refId, score in zip(refIds, scores):
            #print(refId)
            if not refId in self.refToEval:
                self.refToEval[refId] = {}
                self.refToEval[refId]["ref_id"] = refId
            self.refToEval[refId][method] = score

    def setEvalRefs(self):
        self.evalRefs = [eval for refId, eval in self.refToEval.items()]

    def sucess_rate(self):
        sucess = 0
        failure = 0
        self.round_count = {}
        for id, dialog in self.dialog.items():
            if dialog[-1] == 'locate the object':
                sucess += 1
                dialog_len = len(dialog)
                self.round_count[dialog_len] = self.round_count.get(dialog_len, 0) + 1
                #if dialog_len == 6:
                #    print('refer_id:', id)
                #    print('dialog:', dialog)
                #    print('target refer', self.refer[id])
                #print(dialog)
            else:
                failure += 1
        self.sucess = sucess / (sucess + failure)

    def detect_duplicate(self, pred_data):
        pred_removed = {}
        remove_word = 0
        dup_dial_num = 0
        for ref_id, speaker in pred_data.items():
            temp = ''
            temp_list = []
            duplicate = False
            for pred in speaker:
                if temp == '':
                    temp = pred
                else:
                    pred_word = pred.split(' ')
                    temp_word = temp.split(' ')
                    for word in pred_word:
                        if word in temp_word:
                            pred_rmword = pred_word.copy().remove(word)
                            if pred_rmword is None: # duplicate round
                                pred = ''
                                duplicate = True
                            else:
                                pred = ' '.join(pred_rmword)
                            remove_word += 1
                    temp += pred
                temp_list.append(pred)
            pred_removed[ref_id] = temp_list
            if duplicate:
                dup_dial_num += 1
        dup_dial_rate = float(dup_dial_num) / len(pred_data)
        print('remove duplicate word num', remove_word)
        print('dup_dial_rate', dup_dial_rate)
        return pred_removed

    # eval cat in all possible order
    def eval_shuffle(self, template = False):
        pred = {}
        refer = {}
        if template:
            pred_data = self.remove_template(self.pred)
        else:
            pred_data = self.pred

        pred_rm_dup = self.detect_duplicate(pred_data)
        #pred_rm_dup = pred_data

        for ref_id, speaker in pred_rm_dup.items():
            for i, arrange in enumerate(all_possible_arrange[len(speaker)]):
                sent = [' '.join(speaker[j] for j in arrange)]
                pred[str(ref_id) +'_'+ str(i)] = sent
                refer[str(ref_id) +'_'+ str(i)] = self.refer[ref_id]
        return self.re_eval_get_max_avg(pred, refer)

    def re_eval_get_max_avg(self, pred, gt):
        '''
        :param pred: Speakers prediction concate ['s1 s2 s3'] considered all possible order
        :param gt:[RE]
        '''

        #print('tokenization...')
        tokenizer = PTBTokenizer()
        self.pred_tokens = tokenizer.tokenize(pred)
        self.gt_tokens = tokenizer.tokenize(gt)
        # =================================================
        # Set up scorers
        # =================================================
        #print('setting up scorers...')

        # set METEOR as criterion
        scorers = [
            #(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR")#,
            #(Rouge(), "ROUGE_L"),
            #(Cider(), "CIDEr")
        ]
        # =================================================
        # Compute scores
        # =================================================
        eval_result = {}
        for scorer, method in scorers:
            print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.gt_tokens, self.pred_tokens)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    max_score = self.getMaxOrderScore(scs, self.gt_tokens.keys(), m)
                    eval_result[m] = max_score
                    # print ("%s: %0.3f"%(m, sc))

            else:
                max_score = self.getMaxOrderScore(scores, self.gt_tokens.keys(), method)
                eval_result[method] = max_score
                # print ("%s: %0.3f"%(method, score))

        return eval_result

    def getMaxOrderScore(self, scores, refIds, method):

        scores_max = {}
        pred_max_id = {}
        # find max id
        for refId, score in zip(refIds, scores):
            id_origin = refId.split('_')[0]
            if id_origin not in scores_max:
                scores_max[id_origin] = score
                pred_max_id[id_origin] = refId
            elif scores_max[id_origin] < score:
                scores_max[id_origin] = score
                pred_max_id[id_origin] = refId


        max_pred_token_list = {}
        tar_token_list = {}
        for id_o, id_c in pred_max_id.items():
            max_pred_token_list[id_o] = self.pred_tokens[id_c]
            tar_token_list[id_o] = self.gt_tokens[id_c]


        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2"]),# "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        # =================================================
        # Compute scores
        # =================================================
        max_result = {}
        for scorer, med in scorers:

            # print ('computing %s score...'%(scorer.method()))
            score_, scores_ = scorer.compute_score(tar_token_list, max_pred_token_list)
            if type(med) == list:
                for sc, scs, m in zip(score_, scores_, med):
                    #print(np.mean(np.array(scs)), med)
                    max_result[m] = sc
                    if m == method:
                        self.setRefToEvalRefs(scs, tar_token_list.keys(), m)
                        # print ("%s: %0.3f"%(m, sc))

            else:
                max_result[med] = score_
                if med == method:
                    self.setRefToEvalRefs(scores_, tar_token_list.keys(), med)

        return max_result

