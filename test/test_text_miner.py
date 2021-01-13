import dask.dataframe as dd
import pandas as pd
import unittest

from happy_learning.text_miner import TextMiner

DATA_SET: pd.DataFrame = pd.read_csv('data/gun-violence-data_01-2013_03-2018.csv')
TEXT_MINER: TextMiner = TextMiner(df=DATA_SET.iloc[0:1000, ], lang='en', auto_interpret_natural_language=True)


class TextMinerTest(unittest.TestCase):
    """
    Unit test for class TextMiner
    """
    def test_clean(self):
        self.assertTrue(expr=len(DATA_SET['notes'].values[0]) > len(TEXT_MINER.clean_text(phrase=DATA_SET['notes'].values[0])))

    def test_clustering(self):
        pass

    def test_detect_lang(self):
        _lang_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_lang'))
        TEXT_MINER.detect_lang(sampling=True)
        self.assertTrue(expr=_lang_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_lang')) > 0)

    def test_count_occurances(self):
        _occurance_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_shot'))
        TEXT_MINER.count_occurances(features=['notes'], search_text='_shot')
        self.assertTrue(expr=_occurance_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_shot')) > 0)

    def test_generate_linguistic_features(self):
        _ner_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_ner_'))
        _pos_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_pos_'))
        TEXT_MINER.generate_linguistic_features()
        self.assertTrue(expr=_ner_feature == 0 and _pos_feature == 0 and (len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_ner')) > 0) and (len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_pos')) > 0))

    def test_all_generated_features(self):
        TEXT_MINER.generate_linguistic_features()
        TEXT_MINER.count_occurances()
        TEXT_MINER.generate_categorical_features()
        print(TEXT_MINER.get_all_generated_features())
        self.assertTrue(expr=TEXT_MINER.get_all_generated_features().shape[1] > 0)

    def test_get_str_match(self):
        self.assertTrue(expr=len(TEXT_MINER.get_str_match(cases=['abc', 'def'], substring='b')) > 0)

    def test_merge(self):
        _merge_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_merge_'))
        TEXT_MINER.merge(features=['reviewText', 'summary'])
        self.assertTrue(expr=_merge_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_merge_')) > 0)

    def test_replace(self):
        TEXT_MINER.replace(features=['notes'], find_values=['shot'], replace_value='Gianni')
        TEXT_MINER.count_occurances(features=['notes'], search_text='Gianni')
        self.assertTrue(expr=len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='notes_count_Gianni')) > 0)

    def test_splitter(self):
        _split_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_split_'))
        TEXT_MINER.splitter(features=['notes'], sep=' ')
        self.assertTrue(expr=_split_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_split_')) > 0)

    def test_segmentation(self):
        self.assertDictEqual(d1={'enumeration': [],
                                 'phrases': ['notes'],
                                 'id': [],
                                 'email': [],
                                 'rating': [],
                                 'url': [],
                                 'unknown': ['address', 'location']
                                 },
                             d2=TEXT_MINER.segments
                             )

    def test_similarity(self):
        _similarity_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_similarity'))
        TEXT_MINER.similarity()
        self.assertTrue(expr=_similarity_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_similarity')) > 0)

    def test_tfifd(self):
        _tfidf_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_tfidf'))
        TEXT_MINER.tfidf()
        self.assertTrue(expr=_tfidf_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.columns), substring='_tfidf')) > 0)

    def test_translate(self):
        self.assertEqual(first='ciao', second=TEXT_MINER.translate(text='hallo', lang='it').lower())


if __name__ == '__main__':
    unittest.main()
