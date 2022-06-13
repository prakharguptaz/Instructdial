import string
import json
import random
from string import Template
import os
from collections import Counter, defaultdict


def list_tostring(classes):
	assert type(classes) == list
	lenc = len(classes)
	if len(classes)<2:
		return ' '.join(classes)
	elif len(classes)==2:
		return classes[0] + ' and ' + classes[1]
	else:
		return ', '.join(classes[:-1]) + ' and ' + classes[-1]


def get_slot_values(text, slot):
	text_words = text.split()
	slot_words = slot.split()
	slot_inter = []
	word_inter = []
	slots = defaultdict(list)
	for i, (t,s) in enumerate(zip(text_words,slot_words)):
		if "B-" in s:
			if len(slot_inter)>0:
				slot_val = (slot_inter)
				word_val = ' '.join(word_inter)
				# print('word_valb', word_val)
				slots[slot_val[0].replace('I-', '').replace('B-', '')].append(word_val)
				# print(slot_val, word_val)
			slot_inter, word_inter = [], []
			slot_inter.append(s)
			word_inter.append(t)
		elif 'I-' in s:
			slot_inter.append(s)
			word_inter.append(t)			
		elif s=='O':
			if len(slot_inter)>0:
				slot_val = (slot_inter)
				word_val = ' '.join(word_inter)
				# print('word_val', word_val)
				slots[slot_val[0].replace('I-', '').replace('B-', '')].append(word_val)
				# print(slot_val, word_val)
			slot_inter, word_inter = [], []
	if len(slot_inter)>0:
		slot_val = (slot_inter)
		word_val = ' '.join(word_inter)
		slots[slot_val[0].replace('I-', '').replace('B-', '')].append(word_val)
		# print(slot_val, word_val)

	return slots



def get_sequence(dataset_reader, dp, instruction):
	instruction_type = instruction['id']
	definitions = []
	if 'Definition' in instruction:
		definitions = [instruction['Definition']]
	elif 'Definitions' in instruction:
		definitions = instruction['Definitions']
	sequences = []
	if instruction_type == 'intent_classification':
		for definition in definitions:
			num_classes = random.randint(2, len(dataset_reader.intent_classes))
			classes = random.sample(dataset_reader.intent_classes, num_classes)
			mapping = {'classes':list_tostring(classes)}
			mapped_definition = Template(definition).substitute(**mapping)
			text = mapped_definition + ' The response is: ' + dp['response'] + '. The intent is:'
			output = dp['intent_label']
			sequences.append({'text':text, 'output': output})

	if instruction_type == 'emotion_classification':
		for definition in definitions:
			num_classes = random.randint(2, min(len(dataset_reader.intent_classes),20))
			classes = random.sample(dataset_reader.intent_classes, num_classes)
			mapping = {'classes':list_tostring(classes)}
			mapped_definition = Template(definition).substitute(**mapping)
			text = mapped_definition + ' The response is: ' + dp['response'] + '. The emotion is:'
			output = dp['intent_label']
			sequences.append({'text':text, 'output': output})

	if instruction_type == 'slot_tagging':
		for definition in definitions:
			slot_classes = (dataset_reader.slot_classes)
			text, slot = dp['text'], dp['slots']
			slots_dict = get_slot_values(text, slot)
			if 'O' in slots_dict:
				del slots_dict['O']
			# print(slots_dict)
			if len(slots_dict)==0: return []
			mapping = {}
			mapped_definition = Template(definition).substitute(**mapping)
			for k in slots_dict:
				v = random.choice(slots_dict[k])
				text = mapped_definition + ' The response is: ' + dp['text'] +  " [EOS] Question: The value of "+ k +" mentioned in the utterance is"
				output = v
				sequences.append({'text':text, 'output': output})
				# print('ADDING', {'text':text, 'output': output})
	if instruction_type == 'slot_present':
		for definition in definitions:
			slot_classes = (dataset_reader.slot_classes)
			slot_classes = [x.replace('I-', '').replace('B-', '') for x in slot_classes]
			slot_classes = list(set(slot_classes) - set(['[PAD]', 'O']))
			text, slot = dp['text'], dp['slots']
			slots_dict = get_slot_values(text, slot)
			if 'O' in slots_dict:
				del slots_dict['O']
			# print(slots_dict)
			if len(slots_dict)==0: return []
			mapping = {}
			mapped_definition = Template(definition).substitute(**mapping)
			for k in slots_dict:
				v = random.choice(slots_dict[k])
				kval = k.replace('_', ' ')
				text = mapped_definition + ' The response is: ' + dp['text'] +  " [EOS] Question: Is the slot "+ kval +" present in the utterance?"
				output = 'yes'
				sequences.append({'text':text, 'output': output})
				random_slot = None
				while True:
					random_slot = random.choice(slot_classes)
					if random_slot.replace(' ', '-') not in slots_dict:
						break
				text = mapped_definition + ' The response is: ' + dp['text'] +  "[EOS] Question: Is the slot "+ random_slot +" present in the utterance?"
				output = 'no'
				sequences.append({'text':text, 'output': output})
	if instruction_type == 'wow':
		for definition in definitions:
			# import pdb;pdb.set_trace()
			mapping = {}
			mapped_definition = Template(definition).substitute(**mapping)
			context = ' [EOS] '.join(dp['context'])
			knowledge = dp['knowledge'][0]
			response = dp['response']
			text = mapped_definition + ' Dialogue context: ' + context +  " Wikipedia text:  "+ knowledge
			output = response
			sequences.append({'text':text, 'output': output})


	if instruction_type == 'restaurant_8k':
		for definition in definitions:
			slot_classes = (dataset_reader.slot_classes)
			text, slot = dp['text'], dp['slots']
			slots_dict = get_slot_values(text, slot)
			if 'O' in slots_dict:
				del slots_dict['O']
			# print(slots_dict)
			if len(slots_dict)==0: return []
			mapping = {}
			mapped_definition = Template(definition).substitute(**mapping)
			for k in slots_dict:
				v = random.choice(slots_dict[k])
				text = mapped_definition + ' The response is: ' + dp['text'] +  " [EOS] Question: The value of "+ k +" slot is"
				output = v
				sequences.append({'text':text, 'output': output})



	return sequences
