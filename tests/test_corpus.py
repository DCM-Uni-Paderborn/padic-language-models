import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from padic_lm.corpus import Article, articles, last_consumed_row, windows


class CorpusTests(unittest.TestCase):
    def test_article_boundaries_keep_sections_and_exact_source_rows(self):
        rows = ['\n', ' = First = \n', 'alpha', ' == Section == \n', 'beta', ' = Second = \n', 'gamma']
        docs = articles(rows)
        self.assertEqual([(d.start_row, d.stop_row, d.title) for d in docs], [(1,5,'First'),(5,7,'Second')])
        self.assertEqual(docs[0].text, '\n\n'.join(rows[1:5]))
        self.assertNotEqual(docs[0].text_sha256, docs[1].text_sha256)
        with self.assertRaises(ValueError): articles(['unassigned', ' = Title = '])

    def test_old_stream_separator_exclusion_is_conservative(self):
        rows = ['abc', '', 'def']
        self.assertEqual(last_consumed_row(rows, 3), 0)
        self.assertEqual(last_consumed_row(rows, 4), 1)
        self.assertEqual(last_consumed_row(rows, 7), 2)
        self.assertEqual(last_consumed_row(rows, 10), 2)
        for end in (True, 0, 11):
            with self.assertRaises(ValueError): last_consumed_row(rows, end)

    def test_wikitext_spaced_nested_delimiters_are_sections(self):
        rows=[' = Article one = \n','text',' = = Section = = \n','detail',
              ' = = = Subsection = = = \n','more',' = Article two = \n','last']
        docs=articles(rows)
        self.assertEqual([(d.start_row,d.stop_row,d.title) for d in docs],
                         [(0,6,'Article one'),(6,8,'Article two')])
        self.assertIn(' = = Section = = ',docs[0].text)
        self.assertIn(' = = = Subsection = = = ',docs[0].text)

    def test_round_robin_never_crosses_or_overlaps_articles(self):
        docs = [Article(i*10,i*10+10,str(i),'',str(i)) for i in range(3)]
        tokens = [(doc, [i]*9) for i,doc in enumerate(docs)]
        selected = windows(tokens, count=5, length=4, max_per_article=2)
        self.assertEqual([(d.title,start,stop) for d,start,stop,_ in selected],
                         [('0',0,4),('1',0,4),('2',0,4),('0',4,8),('1',4,8)])
        for doc,start,stop,ids in selected:
            self.assertEqual(ids,[int(doc.title)]*4)

    def test_prior_article_and_duplicate_text_exclusions(self):
        docs=[Article(i*10,i*10+10,str(i),'',str(i)) for i in range(4)]
        selected=windows([(doc,list(range(8))) for doc in docs],count=2,length=4,
                         max_per_article=1,exclude_through_row=9,excluded_hashes={'2'})
        self.assertEqual([doc.title for doc,*_ in selected],['1','3'])
        with self.assertRaises(ValueError):windows([(docs[0],[1,2])],count=1,length=4)
        with self.assertRaises(ValueError):windows([],count=True)


if __name__=='__main__': unittest.main()
