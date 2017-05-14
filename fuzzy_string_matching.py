import regex
import string
from fuzzywuzzy import fuzz

def fuzzy_match(s1, s2, max_dist=3, min_proba=75):
    '''
    >>> fuzzy_match('Happy Days', ' happy days ')
    True
    >>> fuzzy_match('happy days', 'sad days')
    False
    >>> fuzzy_match('AB. Habitat', 'AB-Hnbitat')
    True
    >>> fuzzy_match('EXPL SITE RADIOELECTRIQUE', 'EX PL SITE RADIOELECTRIQUE')
    True
    >>> fuzzy_match('AEROPORT DE BORDEAUX', "UK AEROPORT AJ DE BORDEAUX")
    True
    >>> fuzzy_match('FACTURATION COMMERCIALE- RED. POUR EXPL. SITE', 'FACTURE')
    False
    >>> fuzzy_match('DE','FACTURES')
    False
    >>> fuzzy_match('DE', 'A')
    False
    >>> fuzzy_match('DE', 'DEMANDE')
    False

    >>> fuzzy_match('Option GTR 1Oh S1 IMS 20h', 'Oplbn GTR t0h S1 IMS 20h')
    '''
    s1 = normalize(s1)
    s2 = normalize(s2)
    r = regex.compile('({}){{i<=2,d<=2, s<=2, e<={}}}'.format(s1, max_dist))
    ratio = fuzz.ratio(s1, s2)

    #print(r.match(s2))
    #print(ratio)

    if ratio > min_proba:
        return True
    else:
        return False

def normalize(s):
    for p in string.punctuation:
        s = s.replace(p, '')

    return regex.sub('\s+', s.lower().strip(), ' ').replace(' ', '')


def soundex(name, len=4, lang='fr'):
    """ soundex module conforming to Knuth's algorithm
        implementation 2000-12-24 by Gregory Jorgensen
        public domain
    """

    # digits holds the soundex values for the alphabet
    digits_eng = '01230120022455012623010202'
    digits_fr = '01230970072455012683090808'
    if lang == 'fr':
        digits = digits_fr
    else:
        digits = digits_eng
    sndx = ''
    fc = ''

    # translate alpha chars in name to soundex digits
    for c in name.upper():
        if c.isalpha():
            if not fc: fc = c   # remember first letter
            d = digits[ord(c)-ord('A')]
            # duplicate consecutive soundex digits are skipped
            if not sndx or (d != sndx[-1]):
                sndx += d

    # replace first digit with first alpha character
    sndx = fc + sndx[1:]

    # remove all 0s from the soundex code
    sndx = sndx.replace('0','')

    # return soundex code padded to len characters
    return (sndx + (len * '0'))[:len]