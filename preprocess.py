import os
import re
from helloworld import hello
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
        os.chdir(path)
        self.list = os.listdir('.')
        self.removed = False
        self.text = []
        self.a1 = []
        self.a2 = []
        self.names = []
        self.a1splitted = []  # contains list of files which contains list of lines
        self.a2splitted = []  # contains list of files which contains list of lines
        self.pos = [] #contains list of files which containes position of each entity in each line
        self.term_entity = [] # contains list of files which contains list of term number (i.e. T3) and corresponding entity as single string tuples
        self.term_obt = []  # contains list of files which contains list of term number and corresponding OBT reference
        self.do_all_processing()

    def process_file_type(self):
        if not self.removed:
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
            with open(self.a1[i], 'r', encoding='utf-8') as f:
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
                with open(self.a2[i], 'r') as f:
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
                        for k in self.a2splitted[i]:
                            if j[0] in k:
                                self.term_obt[i].append([j[0], k[-1]])

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
                        self.term_entity[i].append([j[0], temp[:-1]]) # saves term number and entity string
                        break

    def do_all_processing(self):
        self.process_file_type()
        self.extract_habitat_info()
        self.process_a1_content()

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



if __name__ == "__main__":
    imported = hello()
    print(imported)
    data = Dataset()

    name, text, pos, term_entity, term_obt = data.get_item_by_name('BB-norm-448557')
    print('types seperated')
