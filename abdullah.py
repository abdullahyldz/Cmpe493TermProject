
from preprocess import Dataset
from packages import Hello
# from pronto import Ontology

if __name__ == '__main__':
    test = Hello.test()
    print(test)
    data = Dataset()
    name, text, pos, term_entity, term_obt = data.get_item_by_name('BB-norm-448557')