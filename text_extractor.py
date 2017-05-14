# coding: utf-8
import bz2
import os
import re
import bs4
import pandas as pd
import csv
import abbyextractor_helper as tools
#from database_helper import get_real_filename
from word import WordFeature
import numpy as np
#XML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)," "))
DATA_DIR = "/home/ahmed/Downloads/BOUYGUES_XML/BOUYGUES_XML_0/"
FILENAME= "1841729699"
FILE = DATA_DIR + FILENAME + "_0001.xml.gz"
file_index=FILENAME+"_001"
class AbbyExtractor:
    pattern_price = re.compile(r"^\-?(\d+,?\d*[,.]?\d*)$")
    pattern_cleanup = re.compile(r"^[^\w.,\-\s\/\%]+$")

    def __init__(self, filename):
        #xmlgz = get_real_filename(filename, page)

        #text_file = bz2.BZ2File(XML_DIR+'/'+filename, 'rb')
        text_file=bz2.BZ2File(filename,'rb')
        content = text_file.read()

        # Parse
        self.soup = bs4.BeautifulSoup(content, 'xml')

    def get_words(self):
        blocks = self.soup.find_all("block", {"blockType": lambda x: x not in ('Separator', 'SeparatorsBox')})

        wrds_blcks = []
        position_char = []
        for i, block in enumerate(blocks):
            if block['blockType'] == 'Table':
                rslt = self._get_words_from_block_table(block)[0]
                pos=self._get_words_from_block_table(block)[1]
            else:
                rslt = self._get_words_from_block_text(block)[0]
                pos=self._get_words_from_block_text(block)[1]
            rslt = self._cleanup_word(rslt)
            if rslt != [[]] and rslt != []:
                wrds_blcks.append(rslt)
                position_char.append(pos)


        df2 =  pd.DataFrame({'page_number':file_index,'positionlrtb': position_char, 'words': wrds_blcks})
        #temp_df.append(df2)
        ##with open('/home/ahmed/internship/cnn_ocr/xml1.csv', 'a') as f:
         #   rows.to_csv(f, header=False)
          #  for row in rows:
           #     df.loc[len(df)] = row
        #df2.to_csv('/home/ahmed/internship/cnn_ocr/try.csv', mode='a', header=False)
        with open('/home/ahmed/internship/cnn_ocr/scale_train/abby_positions.csv', 'a') as f:
            df2.to_csv(f, header=False)
       # with open('/home/ahmed/internship/cnn_ocr/xml1.csv', 'a') as f:
        #    df2.to_csv(f, mode='a', header=False)
        #frames = [temp_df, df2]
        #temp_df=pd.concat(frames)
        #temp_df.to_csv('/home/ahmed/internship/cnn_ocr/xml1.csv',mode ='a', header= False)
       # with open('/home/ahmed/internship/cnn_ocr/xml1.csv', 'a') as f:
        #    (rows).to_csv(f, header=False)
        #fields = pd.DataFrame({'position :  l, r, t, b, area, perimeter': position_char, 'words': wrds_blcks})
        #temp_df = pd.DataFrame({'position :  l, r, t, b, area, perimeter': position_char,'words': wrds_blcks})
        #temp_df.to_csv('/home/ahmed/internship/cnn_ocr/xml.csv')

        #x= temp_df.count()+1
        return wrds_blcks, position_char


    def _get_words_from_block_table(self, block):
        """
        :param block:
        :return: a matrice row x cells (words)
        """
        rows = block.find_all('row')

        block_words = []
        char_pos = []
        for row in rows:
            row_words = self._get_words_from_block_text(row)[0]
            block_words.extend(row_words)
            pos=self._get_words_from_block_text(row)[1]
            char_pos.extend(pos)
        return block_words, char_pos

    def _get_words_from_block_text(self, block):
        """
        :param block:
        :return: a matrice lines x words
        """
        lines = block.find_all('line')
        block_words = []
        char_position =[]
        for line in lines:
            block_words.append(self._get_words_from_line(line)[0])
            char_position.append(self._get_words_from_line(line)[1])
        return block_words, char_position

    def _get_words_from_line(self, line):
        """
        :param line:
        :return: a list of words
        """
        formatting = line.find("formatting")
        charParams = line.find_all('charParams')
        list_words = []
        char_pos=[]
        current_word = None
        for char in charParams:
            if 'wordStart' in char.attrs:
                if char['wordStart'] == '1':
                    # -- check if the last word is saved
                    if current_word != None:
                        list_words.append(current_word)
                    current_word = WordFeature()
                    current_word.fontsize = formatting['fs']

                if current_word != None:
                    if re.match('^[^a-zA-Z0-9.,\-\s\/\%\u00C1-\u00FF]', char.string) is None:
                        l = int(char['l'])
                        r = int(char['r'])
                        t = int(char['t'])
                        b = int(char['b'])
                        #perimeter= 2*(abs(r-l)+ abs(b-t))
                        #area=abs(r-l)*abs(b-t)
                        current_word.add_letter(char.string, l, r, t, b)
                        #char_pos.append(char.string, l, r, t, b)
                        char_pos += [char.string, l, r, t, b]


            else:
                if current_word != None:
                    list_words.append(current_word)
                    current_word = None

        if current_word != None:
            list_words.append(current_word)

        return list_words, char_pos
    def _cleanup_word(self, block):
        for i, line in enumerate(block):
            for j, word_feature in enumerate(line):
                word_feature.word = tools.normalize(word_feature.word)
        return block