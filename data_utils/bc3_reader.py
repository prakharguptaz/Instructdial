import xml.etree.ElementTree as ET
from data_utils.data_reader import Dataset

class BC3Dataset(Dataset):
    def __init__(self, data_path: str, max_seq_length=512, split='train'):
        tree = ET.parse(f'{data_path}/corpus.xml')
        root = tree.getroot()
        data = {}
        for thread in root:
            mails = []
            for elem in thread:
                if elem.tag == 'name':
                    title = elem.text
                elif elem.tag == 'listno':
                    tag = elem.text
                elif elem.tag == 'DOC':
                    for mail_elem in elem:
                        if mail_elem.tag == 'Subject':
                            subject = mail_elem.text
                        elif mail_elem.tag == 'Text':
                            texts = [x.text for x in mail_elem]
                    mails.append({'subject': subject, 'content': texts})
            
            data[tag] = {
                'tag': tag,
                'title': title,
                'mails': mails
            }

        tree = ET.parse(f'{data_path}/annotation.xml')
        root = tree.getroot()

        for thread in root: 
            texts = []
            labels = []
            for elem in thread:
                if elem.tag == 'listno':
                    tag = elem.text
                elif elem.tag == 'annotation':
                    for annotation in elem:
                        if annotation.tag == 'summary':
                            texts.append([x.text for x in annotation])
                        elif annotation.tag == 'labels':
                            labels.append([x.text for x in annotation])
            data[tag]['summary'] = texts
            data[tag]['labels'] = labels
        self.examples = [x for x in data.values()]
        
        num_eval = len(self.examples) // 10
        self.split = split

        if split == 'train':
            self.examples = self.examples[:-2*num_eval]
        elif split == 'dev':
            self.examples = self.examples[-2*num_eval: -num_eval]
        elif split == 'test':
            self.examples = self.examples[-num_eval:]
