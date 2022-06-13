import math
from collections import Counter
from nltk.util import ngrams
import json
import sqlite3
import os
import random
import logging
import ipdb
import sys
import re
from collections import Counter
from nltk.util import ngrams

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")

PATH = '../../datasets/ToD_BERT/MultiWOZ-2.1'
# fin = open('../../datasets/ToD_BERT/MultiWOZ-2.1/mapping.pair')
fin = {"it's": "it is",
       "don't": "do not",
       "doesn't": "does not",
       "didn't": "did not",
       "you'd": "you would",
       "you're": "you are",
       "you'll": "you will",
       "i'm": "i am",
       "they're": "they are",
       "that's": "that is",
       "what's": "what is",
       "couldn't": "could not",
       "i've": "i have",
       "we've": "we have",
       "can't": "cannot",
       "i'd": "i would",
       "aren't": "are not",
       "isn't": "is not",
       "wasn't": "was not",
       "weren't": "were not",
       "won't": "will not",
       "there's": "there is",
       "there're": "there are",
       ". .": ".",
       "restaurants": "restaurant -s",
       "hotels": "hotel -s",
       "laptops": "laptop -s",
       "cheaper": "cheap -er",
       "dinners": "dinner -s",
       "lunches": "lunch -s",
       "breakfasts": "breakfast -s",
       "expensively": "expensive -ly",
       "moderately": "moderate -ly",
       "cheaply": "cheap -ly",
       "prices": "price -s",
       "places": "place -s",
       "venues": "venue -s",
       "ranges": "range -s",
       "meals": "meal -s",
       "locations": "location -s",
       "areas": "area -s",
       "policies": "policy -s",
       "children": "child -s",
       "kids": "kid -s",
       "kidfriendly": "kid friendly",
       "cards": "card -s",
       "upmarket": "expensive",
       "inpricey": "cheap",
       "inches": "inch -s",
       "uses": "use -s",
       "dimensions": "dimension -s",
       "driverange": "drive range",
       "includes": "include -s",
       "computers": "computer -s",
       "machines": "machine -s",
       "families": "family -s",
       "ratings": "rating -s",
       "constraints": "constraint -s",
       "pricerange": "price range",
       "batteryrating": "battery rating",
       "requirements": "requirement -s",
       "drives": "drive -s",
       "specifications": "specification -s",
       "weightrange": "weight range",
       "harddrive": "hard drive",
       "batterylife": "battery life",
       "businesses": "business -s",
       "hours": "hour -s",
       "one": "1",
       "two": "2",
       "three": "3",
       "four": "4",
       "five": "5",
       "six": "6",
       "seven": "7",
       "eight": "8",
       "nine": "9",
       "ten": "10",
       "eleven": "11",
       "twelve": "12",
       "anywhere": "any where",
       "good bye": "goodbye"}

replacements = [(k, v) for k, v in fin.items()]

"""
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))
"""

digitpat = re.compile('\d+')
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")


# FORMAT
# domain_value
# restaurant_postcode
# restaurant_address
# taxi_car8
# taxi_number
# train_id etc..


def prepareSlotValuesIndependent():
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []

    # read databases
    for domain in domains:
        try:
            # fin = file(os.path.join(PATH, 'db/' + domain + '_db.json'))
            fin = open(os.path.join(PATH, domain + '_db.json'), 'r')
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if val == '?' or val == 'free':
                        pass
                    elif key == 'address':
                        dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif key == 'name':
                        dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif key == 'postcode':
                        dic.append((normalize(val), '[' + domain + '_' + 'postcode' + ']'))
                    elif key == 'phone':
                        dic.append((val, '[' + domain + '_' + 'phone' + ']'))
                    elif key == 'trainID':
                        dic.append((normalize(val), '[' + domain + '_' + 'id' + ']'))
                    elif key == 'department':
                        dic.append((normalize(val), '[' + domain + '_' + 'department' + ']'))

                    # NORMAL DELEX
                    elif key == 'area':
                        dic_area.append((normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                    elif key == 'food':
                        dic_food.append((normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                    elif key == 'pricerange':
                        dic_price.append((normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                    else:
                        pass
                    # TODO car type?
        except:
            pass

        if domain == 'hospital':
            dic.append((normalize('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('Hills Road'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('0122324515', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Addenbrookes Hospital'), '[' + domain + '_' + 'name' + ']'))

        elif domain == 'police':
            dic.append((normalize('Parkside'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize('Parkside Police Station'), '[' + domain + '_' + 'name' + ']'))

    # add at the end places from trains
    # fin = open(os.path.join(PATH, 'db/' + 'train' + '_db.json'))
    fin = open(os.path.join(PATH, 'train' + '_db.json'))
    db_json = json.load(fin)
    fin.close()

    for ent in db_json:
        for key, val in ent.items():
            if key == 'departure' or key == 'destination':
                dic.append((normalize(val), '[' + 'value' + '_' + 'place' + ']'))

    # add specific values:
    for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic


def prepareSlotValuesIndependent_mine():
    '''just delex entity names, not price, food or area'''
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []

    normalize_func = normalize_mine
    # read databases
    for domain in domains:
        try:
            # fin = file(os.path.join(PATH, 'db/' + domain + '_db.json'))
            fin = open(os.path.join(PATH, domain + '_db.json'), 'r')
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if val == '?' or val == 'free':
                        pass
                    elif key == 'address':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'address' + ']'))
                    elif key == 'name':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append((normalize_func(val), '[' + domain + '_' + 'name' + ']'))
                    elif key == 'postcode':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'postcode' + ']'))
                    elif key == 'phone':
                        dic.append((val, '[' + domain + '_' + 'phone' + ']'))
                    elif key == 'trainID':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'id' + ']'))
                    elif key == 'department':
                        dic.append((normalize_func(val), '[' + domain + '_' + 'department' + ']'))

                    # NORMAL DELEX
                    # elif key == 'area':
                    #     dic_area.append((normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                    # elif key == 'food':
                    #     dic_food.append((normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                    # elif key == 'pricerange':
                    #     dic_price.append((normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                    # else:
                    #     pass
                    # TODO car type?
        except:
            pass

        if domain == 'hospital':
            dic.append((normalize_func('Hills Rd'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize_func('Hills Road'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize_func('CB20QQ'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223245151', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('0122324515', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize_func('Addenbrookes Hospital'), '[' + domain + '_' + 'name' + ']'))

        elif domain == 'police':
            dic.append((normalize_func('Parkside'), '[' + domain + '_' + 'address' + ']'))
            dic.append((normalize_func('CB11JG'), '[' + domain + '_' + 'postcode' + ']'))
            dic.append(('01223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append(('1223358966', '[' + domain + '_' + 'phone' + ']'))
            dic.append((normalize_func('Parkside Police Station'), '[' + domain + '_' + 'name' + ']'))

    # add at the end places from trains
    # fin = open(os.path.join(PATH, 'db/' + 'train' + '_db.json'))
    # db_json = json.load(fin)
    # fin.close()
    #
    # for ent in db_json:
    #     for key, val in ent.items():
    #         if key == 'departure' or key == 'destination':
    #             dic.append((normalize(val), '[' + 'value' + '_' + 'place' + ']'))
    #
    # # add specific values:
    # for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
    #     dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?

    return utt


def delexicaliseDomain(utt, dictionary, domain):
    for key, val in dictionary:
        if key == domain or key == 'value':
            utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
            utt = utt[1:-1]  # why this?

    # go through rest of domain in case we are missing something out?
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?
    return utt


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    text = re.sub(timepat, ' [value_time] ', text)
    text = re.sub(pricepat, ' [value_price] ', text)
    # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def normalize_for_sql(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    # text = re.sub(timepat, ' [value_time] ', text)
    # text = re.sub(pricepat, ' [value_price] ', text)
    # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    # text = re.sub('^\'', '', text)
    # text = re.sub('\'$', '', text)
    # text = re.sub('\'\s', ' ', text)
    # text = re.sub('\s\'', ' ', text)
    # for fromx, tox in replacements:
    #     text = ' ' + text + ' '
    #     text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    text = text.replace('marys', r"mary''s")
    text = text.replace('restaurant 17', 'restaurant one seven')
    text = text.replace('christ college', r"christ''s college")
    text = text.replace('city centre north bed and breakfast', 'city centre north b and b')
    text = text.replace('cambridge belfry', 'the cambridge belfry')
    text = text.replace('cow pizza kitchen and bar', 'the cow pizza kitchen and bar')
    text = text.replace("peoples portraits exhibition at girton college",
                        r"people''s portraits exhibition at girton college")
    text = text.replace('golden curry', 'the golden curry')
    text = text.replace("shiraz", "shiraz restaurant")
    text = text.replace("queens college", r"queens'' college")
    text = text.replace('alpha milton guest house', 'alpha-milton guest house')
    text = text.replace('cherry hinton village centre', 'the cherry hinton village centre')
    text = text.replace('multiple sports', 'mutliple sports')
    text = text.replace('cambridge chop house', 'the cambridge chop house')
    text = text.replace("cambridge punter", "the cambridge punter")
    text = text.replace("rosas bed and breakfast", r"rosa''s bed and breakfast")
    text = text.replace('el shaddia guesthouse', "el shaddai")
    text = text.replace('swimming pool', 'swimmingpool')
    text = text.replace('night club', 'nightclub')
    text = text.replace("nirala", "the nirala")
    text = text.replace("kings college", r"king''s college")
    text = text.replace('copper kettle', 'the copper kettle')
    text = text.replace('cherry hinton village centre', 'the cherry hinton village centre')
    text = text.replace("kettles yard", r"kettle''s yard")
    text = text.replace("good luck", "the good luck chinese food takeaway")
    text = text.replace("lensfield hotel", "the lensfield hotel")
    text = text.replace("restaurant 2 two", "restaurant two two")
    text = text.replace("churchills college", "churchill college")
    text = text.replace("fitzwilliam museum", "the fitzwilliam museum")
    text = text.replace('cafe uno', 'caffe uno')
    text = text.replace('sheeps green and lammas land park fen causeway',
                        "sheep's green and lammas land park fen causeway")
    text = text.replace("cambridge contemporary art museum", "cambridge contemporary art")
    text = text.replace('graffton hotel restaurant', "grafton hotel restaurant")
    text = text.replace("saint catharine s college", r"saint catharine''s college")
    text = text.replace('meze bar', 'meze bar restaurant')

    return text


def normalize_mine(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    # text = re.sub(timepat, ' [value_time] ', text)
    # text = re.sub(pricepat, ' [value_price] ', text)
    # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    # text = re.sub('[\":\<>@\(\)]', '', text)
    text = re.sub('[\"\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def normalize_lexical(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # # replace time and and price
    # text = re.sub(timepat, ' [value_time] ', text)
    # text = re.sub(pricepat, ' [value_price] ', text)
    # #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    # text = re.sub('[\":\<>@\(\)]', '', text)
    text = re.sub('[\"\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def normalize_beliefstate(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # # replace time and and price
    # text = re.sub(timepat, ' [value_time] ', text)
    # text = re.sub(pricepat, ' [value_price] ', text)
    # #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    # text = re.sub('[\":\<>@\(\)]', '', text)
    text = re.sub('[\"\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            if type(hyps[0]) is list:
                hyps = [hyp.split() for hyp in hyps[0]]
            else:
                hyps = [hyp.split() for hyp in hyps]

            refs = [ref.split() for ref in refs]

            # Shawn's evaluation
            refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class GentScorer(object):
    def __init__(self, detectfile):
        self.bleuscorer = BLEUScorer()

    def scoreBLEU(self, parallel_corpus):
        return self.bleuscorer.score(parallel_corpus)


def sentence_bleu_4(hyp, refs, weights=[0.25, 0.25, 0.25, 0.25]):
    # input : single sentence, multiple references
    count = [0, 0, 0, 0]
    clip_count = [0, 0, 0, 0]
    r = 0
    c = 0

    for i in range(4):
        hypcnts = Counter(ngrams(hyp, i + 1))
        cnt = sum(hypcnts.values())
        count[i] += cnt

        # compute clipped counts
        max_counts = {}
        for ref in refs:
            refcnts = Counter(ngrams(ref, i + 1))
            for ng in hypcnts:
                max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
        clipcnt = dict((ng, min(count, max_counts[ng])) \
                       for ng, count in hypcnts.items())
        clip_count[i] += sum(clipcnt.values())

    bestmatch = [1000, 1000]
    for ref in refs:
        if bestmatch[0] == 0:
            break
        diff = abs(len(ref) - len(hyp))
        if diff < bestmatch[0]:
            bestmatch[0] = diff
            bestmatch[1] = len(ref)
    r = bestmatch[1]
    c = len(hyp)

    p0 = 1e-7
    bp = math.exp(-abs(1.0 - float(r) / float(c + p0)))

    p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 for i in range(4)]
    s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
    bleu_hyp = bp * math.exp(s)

    return bleu_hyp


def remove_model_mismatch_and_db_data(dial_name, target_beliefs, pred_beliefs, domain, t):
    if domain == 'hotel':
        if domain in target_beliefs[t]:
            if 'type' in pred_beliefs[domain] and 'type' in target_beliefs[t][domain]:
                if pred_beliefs[domain]['type'] != target_beliefs[t][domain]['type']:
                    pred_beliefs[domain]['type'] = target_beliefs[t][domain]['type']
            elif 'type' in pred_beliefs[domain] and 'type' not in target_beliefs[t][domain]:
                del pred_beliefs[domain]['type']

    if 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'pizza hut fenditton':
        pred_beliefs[domain]['name'] = 'pizza hut fen ditton'

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'riverside brasserie':
        pred_beliefs[domain]['food'] = "modern european"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'charlie chan':
        pred_beliefs[domain]['area'] = "centre"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'saint johns chop house':
        pred_beliefs[domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'pizza hut fen ditton':
        pred_beliefs[domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'cote':
        pred_beliefs[domain]['pricerange'] = "expensive"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'cambridge lodge restaurant':
        pred_beliefs[domain]['food'] = "european"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'cafe jello gallery':
        pred_beliefs[domain]['food'] = "peking restaurant"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'nandos':
        pred_beliefs[domain]['food'] = "portuguese"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'yippee noodle bar':
        pred_beliefs[domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'copper kettle':
        pred_beliefs[domain]['food'] = "british"

    if domain == 'restaurant' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] in ['nirala', 'the nirala']:
        pred_beliefs[domain]['food'] = "indian"

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'vue cinema':
        if 'type' in pred_beliefs[domain]:
            del pred_beliefs[domain]['type']

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'funky fun house':
        pred_beliefs[domain]['area'] = 'dontcare'

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'little seoul':
        pred_beliefs[domain]['name'] = 'downing college'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'byard art':
        pred_beliefs[domain]['type'] = 'museum'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'trinity college':
        pred_beliefs[domain]['type'] = 'college'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[domain] and pred_beliefs[domain][
        'name'] == 'cambridge university botanic gardens':
        pred_beliefs[domain]['area'] = 'centre'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'lovell lodge':
        pred_beliefs[domain]['parking'] = 'yes'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'whale of a time':
        pred_beliefs[domain]['type'] = 'entertainment'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[domain] and pred_beliefs[domain]['name'] == 'a and b guest house':
        pred_beliefs[domain]['parking'] = 'yes'  # correct name in turn_belief_pred

    if dial_name == 'MUL0116.json' and domain == 'hotel' and 'area' in pred_beliefs[domain]:
        del pred_beliefs[domain]['area']

    return pred_beliefs


class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # hypothesis = [hypothesis]
        # corpus = [corpus]
        # ipdb.set_trace()

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            if type(hyps[0]) is list:
                hyps = [hyp.split() for hyp in hyps[0]]
            else:
                hyps = [hyp.split() for hyp in hyps]
            #
            refs = [ref.split() for ref in refs]
            # hyps = [hyps]
            # hyps = hyps
            # Shawn's evaluation
            # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
            # ipdb.set_trace()
            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class MultiWozDB(object):
    # loading databases
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    dbs = {}
    CUR_DIR = os.path.dirname(__file__)

    for domain in domains:
        db = os.path.join('utils/multiwoz/db/{}-dbase.db'.format(domain))
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs[domain] = c

    def queryResultVenues(self, domain, turn, real_belief=False):
        # query the db
        sql_query = "select * from {}".format(domain)

        if real_belief == True:
            items = turn.items()
        else:
            items = turn['metadata'][domain]['semi'].items()

        flag = True
        for key, val in items:
            if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        try:  # "select * from attraction  where name = 'queens college'"
            return self.dbs[domain].execute(sql_query).fetchall()
        except:
            return []  # TODO test it


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        self.data_name = data_name
        self.slot_dict = prepareSlotValuesIndependent()
        self.delex_dialogues = json.load(open('../../datasets/ToD_BERT/MultiWOZ-2.1/delex_data.json', 'r'))
        self.db = MultiWozDB()
        self.labels = list()
        self.hyps = list()
        # self.venues = json.load(open('../../datasets/ToD_BERT/MultiWOZ-2.1/all_venues.json', 'r'))

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def _parseGoal(self, goal, d, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
        if 'info' in d['goal'][domain]:
            # if d['goal'][domain].has_key('info'):
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in d['goal'][domain]:
                    # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in d['goal'][domain]:
                    # if d['goal'][domain].has_key('reqt'):
                    if 'trainID' in d['goal'][domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in d['goal'][domain]:
                    # if d['goal'][domain].has_key('reqt'):
                    for s in d['goal'][domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in d['goal'][domain]:
                    # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append("reference")

            goal[domain]["informable"] = d['goal'][domain]['info']
            if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                goal[domain]["booking"] = d['goal'][domain]['book']

        return goal

    def _evaluateGeneratedDialogue(self, dialname, dial, goal, realDialogue, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        m_targetutt = [turn['text'] for idx, turn in enumerate(realDialogue['log']) if idx % 2 == 1]

        # pred_beliefs = dial['aggregated_belief']
        pred_beliefs = dial['beliefs']
        target_beliefs = dial['target_beliefs']
        pred_responses = dial['responses']

        for t, (sent_gpt, sent_t) in enumerate(zip(pred_responses, m_targetutt)):
            for domain in goal.keys():

                if '[' + domain + '_name]' in sent_gpt or '_id' in sent_gpt:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION

                        if domain not in pred_beliefs:
                            venues = []
                        else:
                            pred_beliefs = remove_model_mismatch_and_db_data(dialname, target_beliefs, pred_beliefs[t],
                                                                             domain, t)
                            venues = self.db.queryResultVenues(domain, pred_beliefs[t][domain], real_belief=True)

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = venues
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                venue_offered[domain] = venues
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        # if domain + '_reference' in sent_t:
                        #     if 'restaurant_reference' in sent_t:
                        if domain + '_reference' in sent_gpt:
                            if 'restaurant_reference' in sent_gpt:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            # elif 'hotel_reference' in sent_t:
                            elif 'hotel_reference' in sent_gpt:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            # elif 'train_reference' in sent_t:
                            elif 'train_reference' in sent_gpt:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        # if domain + '_' + requestable + ']' in sent_t:
                        if domain + '_' + requestable + ']' in sent_gpt:
                            provided_requestables[domain].append(requestable)

            # print('venues', venue_offered)
            # print('request', provided_requestables)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'info' in realDialogue['goal'][domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # the original method
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         # if realDialogue['goal'][domain].has_key('reqt') and 'id' not in realDialogue['goal'][domain]['reqt']:
            #         if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # Wrong one in HDSA
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         if goal[domain]['requestable'] and 'id' not in goal[domain]['requestable']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, goal[domain]['informable'], real_belief=True)

                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1
            else:
                if domain + '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match) / len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success) / len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats

    def _evaluateRealDialogue(self, dialog, filename):
        """Evaluation of the real dialogue.
        First we loads the user goal and then go through the dialogue history.
        Similar to evaluateGeneratedDialogue above."""
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog['goal'][domain]:
                goal = self._parseGoal(goal, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']

        # iterate each turn
        m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['log']) if idx % 2 == 1]
        for t in range(len(m_targetutt)):
            for domain in domains_in_goal:
                sent_t = m_targetutt[t]
                # for computing match - where there are limited entities
                if domain + '_name' in sent_t or '_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        venues = self.db.queryResultVenues(domain, dialog['log'][t * 2 + 1])

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                for requestable in requestables:
                    # check if reference could be issued
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                                    # return goal, 0, match, real_requestables
                            elif 'train_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable in sent_t:
                            provided_requestables[domain].append(requestable)

        # offer was made?
        for domain in domains_in_goal:
            # if name was provided for the user, the match is being done automatically
            # if dialog['goal'][domain].has_key('info'):
            if 'info' in dialog['goal'][domain]:
                # if dialog['goal'][domain]['info'].has_key('name'):
                if 'name' in dialog['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        # HARD (0-1) EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match, success = 0, 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, dialog['goal'][domain]['info'], real_belief=True)
                # print(goal_venues)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1

            else:
                if domain + '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if match == len(goal.keys()):
            match = 1
        else:
            match = 0

        # SUCCESS
        if match:
            for domain in domains_in_goal:
                domain_success = 0
                success_stat = 0
                if len(real_requestables[domain]) == 0:
                    # check that
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if success >= len(real_requestables):
                success = 1
            else:
                success = 0

        return goal, success, match, real_requestables, stats

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities

    def evaluate(self, dialogue, filename, real_dialogues=False, mode='valid'):
        """Gathers statistics for the whole sets."""
        delex_dialogues = self.delex_dialogues
        successes, matches = 0, 0
        total = 0

        gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                         'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        """
                for idx, (filename, dial) in enumerate(dialogues.items()):
                    data = delex_dialogues[filename]

                    goal, success, match, requestables, _ = self._evaluateRealDialogue(data, filename)

                    success, match, stats = self._evaluateGeneratedDialogue(filename, dial, goal,
                                                                                 data, requestables,
                                                                                 soft_acc=mode == 'soft')
        """

        data = delex_dialogues[filename]
        goal, success, match, requestables, _ = self._evaluateRealDialogue(data, filename)
        success, match, stats = self._evaluateGeneratedDialogue(dialogue,
                                                                goal,
                                                                data,
                                                                requestables,
                                                                soft_acc=mode == 'soft')

        successes += success
        matches += match
        total += 1

        for domain in gen_stats.keys():
            gen_stats[domain][0] += stats[domain][0]
            gen_stats[domain][1] += stats[domain][1]
            gen_stats[domain][2] += stats[domain][2]

        if 'SNG' in filename:
            for domain in gen_stats.keys():
                sng_gen_stats[domain][0] += stats[domain][0]
                sng_gen_stats[domain][1] += stats[domain][1]
                sng_gen_stats[domain][2] += stats[domain][2]

        if real_dialogues:
            # BLUE SCORE
            corpus = []
            model_corpus = []
            bscorer = BLEUScorer()

            """
            for dialogue in dialogues:
                data = real_dialogues[dialogue]
                model_turns, corpus_turns = [], []
                # for idx, turn in enumerate(data['sys']):
                for idx, turn in enumerate(data):
                    corpus_turns.append([turn])
                for turn in dialogues[dialogue]['responses']:
                    model_turns.append([turn])

                # ipdb.set_trace()
                if len(model_turns) == len(corpus_turns):
                    corpus.extend(corpus_turns)
                    model_corpus.extend(model_turns)
                else:
                    raise ('Wrong amount of turns')
            """

            data = real_dialogues[dialogue]
            model_turns, corpus_turns = [], []
            # for idx, turn in enumerate(data['sys']):
            for idx, turn in enumerate(data):
                corpus_turns.append([turn])
            for turn in dialogue['responses']:
                model_turns.append([turn])

                # ipdb.set_trace()
            if len(model_turns) == len(corpus_turns):
                corpus.extend(corpus_turns)
                model_corpus.extend(model_turns)
            else:
                raise ('Wrong amount of turns')

            model_corpus_len = []
            for turn in model_corpus:
                if turn[0] == '':
                    model_corpus_len.append(True)
                else:
                    model_corpus_len.append(False)
            if all(model_corpus_len):
                print('no model response')
                model_corpus = corpus
            # ipdb.set_trace()
            blue_score = bscorer.score(model_corpus, corpus)
        else:
            blue_score = 0.

        report = ""
        report += '{} Corpus Matches : {:2.2f}%'.format(mode, (matches / float(total) * 100)) + "\n"
        report += '{} Corpus Success : {:2.2f}%'.format(mode, (successes / float(total) * 100)) + "\n"
        report += '{} Corpus BLEU : {:2.4f}%'.format(mode, blue_score) + "\n"
        report += 'Total number of dialogues: %s ' % total

        print(report)

        return report, successes / float(total), matches / float(total)


def postprocess_gpt2(generated_raw_data):
    generated_proc_data = {}
    for key, value in generated_raw_data.items():
        target_beliefs = value['target_turn_belief']
        target_beliefs_dict = []
        beliefs = value['generated_turn_belief']
        belief_dict = []

        for turn_bs in beliefs:
            bs_dict = {}
            for bs in turn_bs:
                if len(bs.split()) < 3:
                    continue
                if bs in ['', ' ']:
                    continue
                domain = bs.split()[0]
                if domain not in ['train', 'taxi', 'hotel', 'hospital', 'attraction', 'restaurant']:
                    print(key, domain)
                    continue
                if 'book' in bs:
                    continue
                slot = bs.split()[1]
                val = ' '.join(bs.split()[2:])
                if val == 'none':
                    continue
                if domain not in bs_dict:
                    bs_dict[domain] = {}
                bs_dict[domain][slot] = val
            belief_dict.append(bs_dict)

        aggregated_belief_dict = {}
        for bs in value['generated_belief']:
            if len(bs.split()) < 3:
                # print('skipping {}'.format(bs))
                continue
            domain = bs.split()[0]
            if domain not in ['train', 'taxi', 'hotel', 'hospital', 'attraction', 'restaurant']:
                print(domain)
                continue
            if 'book' in bs:
                continue
            slot = bs.split()[1]
            val = ' '.join(bs.split()[2:])
            if val == 'none':
                continue
            if domain not in aggregated_belief_dict:
                aggregated_belief_dict[domain] = {}
            aggregated_belief_dict[domain][slot] = val

        for turn_bs in target_beliefs:
            bs_dict = {}
            for bs in turn_bs:
                if bs in ['', ' ']:
                    continue
                domain = bs.split()[0]
                if domain not in ['train', 'taxi', 'hotel', 'hospital', 'attraction', 'restaurant']:
                    print(domain)
                    continue
                if 'book' in bs:
                    continue
                slot = bs.split()[1]
                val = ' '.join(bs.split()[2:])
                if val == 'none':
                    continue
                if domain not in bs_dict:
                    bs_dict[domain] = {}
                bs_dict[domain][slot] = val
            target_beliefs_dict.append(bs_dict)

        if aggregated_belief_dict != belief_dict[-1]:
            for domain in aggregated_belief_dict:
                if domain == 'attraction' and domain in belief_dict[-1] and len(
                        aggregated_belief_dict[domain].keys()) < len(belief_dict[-1][domain].keys()):
                    aggregated_belief_dict[domain] = belief_dict[-1][domain]
                elif domain == 'restaurant' and domain in aggregated_belief_dict and 'name' in aggregated_belief_dict[
                    domain] and aggregated_belief_dict[domain]['name'] == 'lovell lodge' and domain in belief_dict[
                    -1] and 'name' in belief_dict[-1][domain] and belief_dict[-1][domain]['name'] == 'restaurant 17':
                    # ipdb.set_trace()
                    aggregated_belief_dict[domain] = belief_dict[-1][domain]
                elif domain == 'restaurant' and 'name' in aggregated_belief_dict[domain] and \
                        aggregated_belief_dict[domain]['name'] == 'curry garden' and 'area' in aggregated_belief_dict[
                    domain] and aggregated_belief_dict[domain]['area'] == 'east' and domain in belief_dict:
                    aggregated_belief_dict[domain] = belief_dict[-1][domain]

        generated_proc_data[key] = {
            'name': key,
            'responses': value['generated_response'],
            'beliefs': belief_dict,
            'aggregated_belief': aggregated_belief_dict,
            'target_beliefs': target_beliefs_dict,
            'generated_action': value['generated_action'],
            'target_action': value['target_action'],
        }
    return generated_proc_data
