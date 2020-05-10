import os
import re
from pronto import Ontology
from nltk.corpus import stopwords
import numpy as np
import nltk
from gensim.models import Word2Vec
'''
 Bazı dosyalarda hiç habitat örneği yok, şimdilik onları listeden çıkarmadım. 
 
 İleride PyTorch Dataset classına dönüştürülebilecek şekilde yazmaya çalıştım
 
 Development set ve test set için de ufak değişikliklerle kullanılabilir
 
 TO DO:
 1) Şu an için case-folding yapmadım ama kolayca ekleyebilirim.
    lower() and casefold() will convert the string to lowercase, 
    but casefold() will convert even the caseless letters such as the ß in German to ss.
 2) DONE - Encoding sorunu olan dosyaları nasıl açacağımı öğrenicem. 
 3) Birden çok pozisyon bilgisini nasıl daha kullanışlı kodlayabilirim ona bakıcam. (Bazılarının pozisyon
 bilgisi 697 722;729 730 şu şekilde verilmiş, şu anda 4ünü de tek bir pozisyon listesinde tutuyorum, ayrı iki örnek oluşturabilirim.)
 '''


class Dataset:
    def __init__(self, path='./train/'):
        self.path = path
        if 'train' in self.path: # to differentiate if the set is training or others
            self.train = True
        else:
            self.train = False
        self.list = os.listdir(path)
        self.removed = False
        self.text = []
        self.a1 = []
        self.a2 = []
        self.names = []
        self.a1splitted = []  # contains list of files which contains list of lines
        self.a2splitted = []  # contains list of files which contains list of lines
        self.pos = []  # contains list of files which containes position of each entity in each line
        self.term_entity = []  # contains list of files which contains list of term number (i.e. T3) and corresponding entity as single string tuples
        self.term_obt = []  # contains list of files which contains list of term number and corresponding OBT reference
        self.onto = Ontology('./OntoBiotope.obo')
        self.vocab = []
        self.term_list = []
        self.abbv = get_acronym_list()
        self.do_all_processing()

    def process_file_type(self):
        if (os.path.isfile(self.path+'LICENSE') and os.path.isfile(self.path+'README')):
            self.list.remove('LICENSE')
            self.list.remove('README')
        self.removed = True
        for i in self.list:
            type = i.split(".")
            if type[1] == 'a1':
                self.a1.append(i)
            elif type[1] == 'a2':
                self.a2.append(i)
            else:
                self.text.append(i)
                self.names.append(type[0])

    def extract_habitat_info(self):
        if not self.removed:
            self.process_file_type()
        for i in range(len(self.text)):
            self.a1splitted.append([])
            self.a2splitted.append([])
            self.term_obt.append([])
            read1 = True
            read2 = True
            with open(self.path + self.a1[i], 'r', encoding='utf-8') as f:
                try:
                    a1corpus = f.read().split('\n')
                except UnicodeDecodeError:
                    # 2 dosyada beta gibi yunan harfleri encoding hatası veriyor, onları şimdilik eklemiyorum.
                    print(self.names[i])
                    read1 = False
            if read1:
                for k in a1corpus:
                    self.a1splitted[i].append(re.split("\t|;| ", k))
                j = 0
                while j < len(self.a1splitted[i]):
                    if "Habitat" not in self.a1splitted[i][j]:
                        self.a1splitted[i].remove(self.a1splitted[i][j])
                        j = j-1
                    j = j+1
                if len(self.a2) > 0:
                    with open(self.path + self.a2[i], 'r') as f:
                        try:
                            a2corpus = f.read().split('\n')
                        except UnicodeDecodeError:
                            print(self.names[i])
                            read2 = False
                            self.a1splitted.remove(self.a1splitted[i])
                    if read2:
                        for k in a2corpus:
                            self.a2splitted[i].append(re.split("\t|:| ", k))
                        for j in self.a1splitted[i]:
                            once = False
                            for k in self.a2splitted[i]:
                                if j[0] in k and not once:
                                    self.term_obt[i].append([j[0], k[-1]])
                                    once = True


    def process_a1_content(self):
        for i in range(len(self.a1splitted)): # each file
            self.pos.append([])
            self.term_entity.append([])
            for j in self.a1splitted[i]: # each line
                for k in range(2, len(j)):
                    try:
                        j[k] = int(j[k])
                    except ValueError: # entity starts
                        self.pos[i].append(j[2:k])  # saves position info
                        temp = ''
                        for l in range(k, len(j)):
                            temp = temp + j[l] + ' '
                        self.term_entity[i].append([j[0], temp[:-1]])  # saves term number and entity string
                        break

    def create_term_list(self):
        for i in range(len(self.term_entity)):
            self.term_list.append(self.term_entity[i][1])

    def do_all_processing(self):
        self.process_file_type()
        self.extract_habitat_info()
        self.process_a1_content()
        if self.train:
            self.create_vocabulary()

    def get_item_by_index(self, index): # get all attributes of the file by its index
        name = self.names[index]
        pos = self.pos[index]
        term_entity = self.term_entity[index]
        term_obt = self.term_obt[index]
        text = self.text[index]
        return name, text, pos, term_entity, term_obt

    def get_item_by_name(self, name): # get all attributes of the file by its name
        try:
            index = self.names.index(name)
        except ValueError:
            print("No item with given name")
            return
        return self.get_item_by_index(index)

    def exact_match(self, term):  # if a given term exists in the set, return its obt
        found = False
        obt = None

        lemma_term = lemmatize_term(term)
        if lemma_term == term:
            terms = [term]
        else:
            terms = [term, lemma_term]
        for j in terms:
            for i in self.onto.terms:  # search the given term in ontology
                onto_t = self.onto[i]
                if onto_t.name == j:
                    found = True
                    temp = onto_t.id
                    obt = temp[4:]
                    return found, obt

        for i in range(len(self.term_entity)):  # search given term in training set and if found retrieve its obt
            for j in range(len(self.term_entity[i])):
                iterate = self.term_entity[i][j][1]
                if iterate == term:
                    obt = self.term_obt[i][j][1]
                    found = True
                    return found, obt
        return found, obt

    def get_number_of_terms(self): # unnecessary, but let's keep it for now
        total = 0
        for i in self.term_entity:
            total = total + len(i)
        return total

    def create_vocabulary(self):
        for i in self.term_entity:
            for j in i:
                words = remove_stop_words(j[1])
                for k in words:
                    if k not in self.vocab:
                        self.vocab.append(k)

    def get_term_entity_list(self):  # extends term-entity tuples into a single list
        entity_list = []
        for i in self.term_entity:
            entity_list.extend(i)
        return entity_list

    def get_term_obt_list(self): # extends term-obt tuples into a single list
        obt_list = []
        for i in self.term_obt:
            obt_list.extend(i)
        return obt_list

def remove_stop_words(term):
    stop = stopwords.words('english')
    words = term.split()
    k = 0
    while k < len(words):
        if words[k] in stop:
            words.remove(words[k])
            k = k - 1
        k = k + 1
    return words

def lemmatize_term(term):
    words = term.split()
    lemma = nltk.wordnet.WordNetLemmatizer()
    new_words = []
    for i in words:
        new_words.append(lemma.lemmatize(i))
    new_term = " ".join(new_words)
    return new_term

def cos_similarity(str1, str2, vocab): # given two strings and vocabulary, computes cosine distance
    term1 = np.zeros([len(vocab),])
    term2 = np.zeros([len(vocab),])
    words1 = remove_stop_words(str1)
    words2 = remove_stop_words(str2)
    for i in words1:
        if i in vocab:
            ind = vocab.index(i)
            term1[ind] = 1
    for i in words2:
        if i in vocab:
            ind = vocab.index(i)
            term2[ind] = 1
    norm = np.linalg.norm(term1)*np.linalg.norm(term2)
    if norm != 0:
        similarity = np.dot(term1, term2)/norm
    else:
        similarity = 0
    return similarity

def most_similar_obt(train_set, term):  # given a term from test, finds the most similar word in terms of cosine similarity
    entity_list = train_set.get_term_entity_list()
    obt_list = train_set.get_term_obt_list()
    similarity_res = np.zeros([len(entity_list), ])
    for i in range(len(entity_list)):
        train_term = entity_list[i][1]
        similarity_res[i] = cos_similarity(train_term, term, train_set.vocab)

    ind = np.argmax(similarity_res)
    obt = obt_list[ind][1]
    return obt


def results(train_set, test_set): # iterates through test set terms and creates a list of estimated OBTs
    estimated_obts = []
    for i in test_set.term_entity:
        for j in i:
            term = find_process_abbreviation(j[1], test_set.abbv)
            exact, obt = train_set.exact_match(term)
            if exact:
                estimated_obts.append(obt)
            else:
                obt = most_similar_obt(train_set, term)
                estimated_obts.append(obt)
    obts = test_set.get_term_obt_list()
    test_terms = test_set.get_term_entity_list()
    train_terms = train_set.get_term_entity_list()
    count = 0
    error_list = [['Input Term', 'Normalized Term', 'Normalized OBT', 'True Normalized Term', 'True Normalized OBT']]
    for i in range(len(estimated_obts)):
        if estimated_obts[i] == obts[i][1]:
            count = count + 1
        else:
            error_list.append([test_terms[i][1], train_set.onto['OBT:' + estimated_obts[i]].name, estimated_obts[i], train_set.onto['OBT:'+obts[i][1]].name, obts[i][1]])
    success = count/len(estimated_obts)
    with open('./errorlog.txt', 'w') as f:
        for i in error_list:
            f.write('{} || {} || {} || {} || {}\n'.format(i[0], i[1], i[2], i[3], i[4]))
    return success

def get_acronym_list(path='./acronym_set.txt'):
    with open(path, 'r') as f:
        acr_lines = f.read().split('\n')
    seperated = []
    for i in acr_lines:
        temp = re.split(":", i)
        if len(temp) > 1:
            temp[0] = temp[0][:-1]
            temp[1] = temp[1][1:]
            seperated.extend(temp)
    return seperated

def find_process_abbreviation(term, acr_list):
    words = term.split()
    for i in range(len(words)):
        if words[i].isupper():
            words[i] = process_abbv(words[i], acr_list)
    new_term = " ".join(words)
    return new_term

def process_abbv(word, acr_list):
    try:
        ind = acr_list.index(word)
        word = acr_list[ind+1]
    except ValueError:
        return word
    return word

def success_rate(train_set, test_set): # deprecated
    count = 0
    for i in test_set.term_entity:
        for j in i:
            iterate = j[1]
            term = train_set.get_ontology_term(iterate)
            if term != None:
                count = count + 1
    success = count/test_set.get_number_of_terms()
    return success




if __name__ == "__main__":

    dev_set = Dataset(path='./dev/')
    train_set = Dataset(path='./train/')
    result = results(train_set, dev_set)
    ''''
    name, text, pos, term_entity, term_obt = data.get_item_by_index(5)
    term_name = data.get_ontology_term('gastric mucosa')
    '''

    #print(result)
