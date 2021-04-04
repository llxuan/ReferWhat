import nltk

Synonym = [['man', 'person', 'guy', 'male',  'palyer'],
           ['boy','kid', 'child'],
           ['girl','kid','child'],
           ['catcher', 'player', 'guy', 'person'],
           ['woman', 'person', 'guy',  'female', 'lady'],
           ['slider', 'skier', 'person', 'guy'],
           ['center', 'middle','mid', 'central', 'centre'],
           ['shirt','tshirt','Tshirt','t-shirt','T-shirt','teeshirt'],
           ['talking','speaking'],
           ['taxi', 'cab', 'car', 'trunk', 'vehicle'],
           ['desk', 'table', 'deck'],
           ['room', 'bedroom'],
           ['blue', 'bluish'],
           ['yellow', 'yellowish'],
           ['green', 'greenish'],
           ['black', 'dark', 'darker'],
           ['grey','gray'],
           ['brown','brownish'],
           ['laptop','computer'],
           ['bike', 'bicycle', 'motorcycle'],
           ['big', 'large','huge'],
           ['small', 'little', 'tiny'],
           ['first', '1st'],
           ['second', '2nd'],
           ['third', '3rd'],
           ['forth', '4th'],
           ['giraffe', 'giraffa', 'giraffee'],
           ['glass', ' cup'],
           ['donut', 'doughnut'],
           ['umpire', 'ump']
]

class Simulator(object):
    # the REUer simulator
    def __init__(self):
        self.synonym = Synonym

    def respond(self, re_entities, context):
        ans = 'cannot locate the object'
        max_match_num = 0
        for re_entity in re_entities:
            re_entity, remove_list_len = self.compare(re_entity.copy(), context)
            if max_match_num < remove_list_len:
                max_match_num = remove_list_len
            if re_entity == []:
                ans = 'locate the object'
                break
        return ans, max_match_num

    def compare(self, entities, context):
        remove_list = []
        for keyword in entities:
            for sent in context:
                if self.find_synonym(keyword, sent):
                    if keyword not in remove_list:
                        remove_list.append(keyword)
        for word in remove_list:
            entities.remove(word)
        return entities, len(remove_list)

    def find_synonym(self, keyword, sent):
        sent_token = nltk.tokenize.word_tokenize(sent)
        for word in sent_token:
            if word == keyword:
                return True
            else:
                for syn_word_list in self.synonym:
                    if keyword in syn_word_list and word in syn_word_list:
                        return True
        else:
            return False