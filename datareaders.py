import argparse
import json
import logging
import numpy as np
import os
import random
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange

import settings
from constants import SPECIAL_TOKENS

from data_utils import atis_reader, cqa_reader, dialoglue_reader, dnli_reader, multiwoz_reader, decode_reader, \
    dialfact_reader
from data_utils import eval_reader, hf_reader, otters_reader, snips_reader, wow_reader, qaconv_reader, bc3_reader, \
    topicalchat_reader, spolin_reader

from data_utils import qmsum_reader, dialogsum_reader, mutual_reader, timedial_reader, star_reader
from data_utils import deal_reader, casino_reader, empathy_reader, airdialoglue_reader, reddit_advice_reader, \
    dialogre_reader
from data_utils import persuasion_reader, woz_reader, chitchat_reader, doc2dial_reader, emotionlines_reader
from data_utils import goemotions_reader, cider_reader, tod_reader, opendialkg_reader, me2e_reader, flodial_reader
from data_utils import toxichat_reader, bad_reader, buildbreakfix_reader, saferdialogues_reader, gensf_reader
from data_utils import kvret_reader, camrest676_reader, frames_reader, schemaguided_reader
from data_utils import wozdst_reader, msre2edst_reader, taskmasterdst_reader, multiwozdst_reader


def get_reader(args, dataset):
    # Data readers
    config = json.load(open(args.configfile, 'r'))

    datasetconfig = config.get(dataset, None)
    if datasetconfig is None and 'dataset_configs' in config and config['dataset_configs'].get(dataset,
                                                                                               None) is not None:
        datasetconfig = config['dataset_configs'].get(dataset, None)
        ## TODO add logger
        print('Found dataset config', datasetconfig)

    if 'intent-clinc' in dataset:
        token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
        dataset_reader = dialoglue_reader.IntentDataset(settings.DIALOGUE_PATH + datasetconfig['train_data_path'],
                                                        datasetconfig['max_seq_length'], token_vocab_name)

    if 'intent-banking' in dataset:
        token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
        dataset_reader = dialoglue_reader.IntentDataset(settings.DIALOGUE_PATH + datasetconfig['train_data_path'],
                                                        datasetconfig['max_seq_length'], token_vocab_name)

    if 'intent-hwu' in dataset:
        token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
        dataset_reader = dialoglue_reader.IntentDataset(settings.DIALOGUE_PATH + datasetconfig['train_data_path'],
                                                        datasetconfig['max_seq_length'], token_vocab_name)

    if dataset == 'slot-restaurant8k':
        token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
        dataset_reader = dialoglue_reader.SlotDataset(settings.DIALOGUE_PATH + datasetconfig['train_data_path'],
                                                      datasetconfig['max_seq_length'],
                                                      token_vocab_name)

    if dataset == 'slot-dstc8_sgd':
        token_vocab_name = os.path.basename(datasetconfig['token_vocab_path']).replace(".txt", "")
        dataset_reader = dialoglue_reader.SlotDataset(settings.DIALOGUE_PATH + datasetconfig['train_data_path'],
                                                      datasetconfig['max_seq_length'],
                                                      token_vocab_name)

    if dataset == 'wow':
        dataset_reader = wow_reader.WoWDataset(settings.WOW_PATH + datasetconfig['train_data_path'],
                                               datasetconfig['max_seq_length'])

    if dataset == 'topicalchat':
        dataset_reader = topicalchat_reader.TopicalChatDataset(split=datasetconfig['split'],
                                                               type_seen=datasetconfig['type_seen'])

    if dataset == 'otters':
        dataset_reader = otters_reader.OttersDataset(split=datasetconfig['split'])

    if dataset == 'spolin':
        dataset_reader = spolin_reader.SpolinDataset(data_path=settings.SPOLIN_PATH, split=datasetconfig['split'])

    if dataset == 'dnli':
        dataset_reader = dnli_reader.DNLIDataset(split=datasetconfig['split'])

    if dataset == 'decode':
        dataset_reader = decode_reader.DECODEDataset(split=datasetconfig['split'])

    if dataset == 'dialfact':
        dataset_reader = dialfact_reader.DialfactDataset(split=datasetconfig['split'])

    if dataset == 'dialogre':
        dataset_reader = dialogre_reader.DialogREDataset(data_path=settings.DIALOGRE_PATH, split=datasetconfig['split'])

    if dataset == 'dailydialog':
        dataset_reader = hf_reader.DailyDialogDataset(split=datasetconfig['split'])

    if dataset == 'empathetic_dialogues':
        dataset_reader = hf_reader.EmpatheticDialoguesDataset(split=datasetconfig['split'])

    if dataset == 'convai2':
        dataset_reader = hf_reader.Convai2Dataset(split=datasetconfig['split'])

    if dataset == 'personachat':
        dataset_reader = hf_reader.PersonachatDataset(split=datasetconfig['split'])

    if dataset == 'top':
        dataset_reader = dialoglue_reader.TOPDataset(settings.DIALOGUE_PATH, datasetconfig,
                                                     split=datasetconfig['split'], )

    if dataset == 'atis':
        dataset_reader = atis_reader.AtisDataset(settings.ATIS_PATH, split=datasetconfig['split'], )

    if dataset == 'snips':
        dataset_reader = snips_reader.SnipsDataset(settings.SNIPS_PATH, split=datasetconfig['split'])

    if dataset == 'coqa':
        dataset_reader = cqa_reader.CQADataset(settings.COQA_PATH, 'coqa', split=datasetconfig['split'])
    if dataset == 'quac':
        dataset_reader = cqa_reader.CQADataset(settings.QUAC_PATH, 'quac', split=datasetconfig['split'])
    if dataset == 'qaconv':
        dataset_reader = qaconv_reader.QAConvDataset(settings.QACONV_PATH, split=datasetconfig['split'])
    if dataset == 'bc3':
        dataset_reader = bc3_reader.BC3Dataset(settings.BC3_PATH, split=datasetconfig['split'])
    if dataset == 'qmsum':
        dataset_reader = qmsum_reader.QMSumDataset(settings.QMSUM_PATH, split=datasetconfig['split'])
    if dataset == 'dialogsum':
        dataset_reader = dialogsum_reader.DialogSumDataset(settings.DIALOGUESUM_PATH, split=datasetconfig['split'])
    if dataset == 'samsum':
        dataset_reader = hf_reader.SamsumDataset(split=datasetconfig['split'])
    if dataset == 'mutual':
        dataset_reader = mutual_reader.MutualDataset(settings.MUTUAL_PATH, split=datasetconfig['split'])
    if dataset == 'timedial':
        dataset_reader = hf_reader.TimeDialDataset(split=datasetconfig['split'])
        # dataset_reader = timedial_reader.TimeDialDataset(settings.TIMEDIAL_PATH)

    if dataset == 'star':
        dataset_reader = star_reader.STARDataset(datasetconfig['train_data_path'], datasetconfig['max_seq_length'],
                                                 datasetconfig['token_vocab_path'], split=datasetconfig['split'])

    if dataset == 'airdialogue':
        dataset_reader = airdialoglue_reader.AirDialogueDataset(settings.AIRDIALOGLUE_PATH,
                                                                split=datasetconfig['split'])
    if dataset == 'deal':
        dataset_reader = deal_reader.DealDataset(settings.DEAL_PATH, split=datasetconfig['split'])

    if dataset == 'casino':
        dataset_reader = casino_reader.CasinoDataset(settings.CASINO_PATH, split=datasetconfig['split'])

    if dataset == 'empathy':
        dataset_reader = empathy_reader.EmpathyDataset(settings.EMPATHY_PATH, split=datasetconfig['split'])

    if dataset == 'reddit-advice':
        dataset_reader = reddit_advice_reader.RedditAdviceDataset(settings.REDDIT_ADVICE_PATH,
                                                                  split=datasetconfig['split'])

    if dataset == 'persuasion':
        dataset_reader = persuasion_reader.PersuasionDataset(settings.PERSUASION_PATH, split=datasetconfig['split'])

    if dataset == 'eval':
        dataset_reader = eval_reader.EvalDataset(settings.EVAL_PATH, split=datasetconfig['split'])

    if dataset == 'chitchat':
        dataset_reader = chitchat_reader.ChitChatDataset(settings.CHITCHAT_PATH, split=datasetconfig['split'])

    if dataset == 'doc2dial':
        dataset_reader = doc2dial_reader.Doc2DialDataset(settings.DOC2DIAL_PATH, split=datasetconfig['split'])

    if dataset == 'emotionlines':
        dataset_reader = emotionlines_reader.EmotionLinesDataset(settings.EMOTIONLINES_PATH,
                                                                 split=datasetconfig['split'])
    if dataset == 'goemotions':
        dataset_reader = goemotions_reader.GoEmotionsDataset(settings.GOEMOTIONS_PATH, split=datasetconfig['split'])

    if dataset == 'cider':
        dataset_reader = cider_reader.CiderDataset(settings.CIDER_PATH, split=datasetconfig['split'])

    if dataset == 'multiwoz':
        dataset_reader = tod_reader.TodDataset(settings.TOD_PATH, 'multiwoz', split=datasetconfig['split'])

    if dataset == 'camrest676':
        dataset_reader = camrest676_reader.Camrest676Dataset(settings.CAMREST676_PATH, split=datasetconfig['split'])

    if dataset == 'woz':
        dataset_reader = woz_reader.WozDataset(settings.WOZ_PATH, split=datasetconfig['split'])
        # dataset_reader = tod_reader.TodDataset(settings.TOD_PATH, 'woz', split=datasetconfig['split'])

    if dataset == 'smd':
        dataset_reader = tod_reader.TodDataset(settings.TOD_PATH, 'smd', split=datasetconfig['split'])

    if dataset == 'frames':
        dataset_reader = frames_reader.FramesDataset(settings.FRAMES_PATH, split=datasetconfig['split'])

    if dataset == 'msre2e':
        dataset_reader = me2e_reader.Me2eDataset(settings.ME2E_PATH, split=datasetconfig['split'])

    if dataset == 'taskmaster':
        dataset_reader = tod_reader.TodDataset(settings.TOD_PATH, 'taskmaster', split=datasetconfig['split'])

    if dataset == 'metalwoz':
        dataset_reader = tod_reader.TodDataset(settings.TOD_PATH, 'metalwoz', split=datasetconfig['split'])

    if dataset == 'schema':
        dataset_reader = schemaguided_reader.SchemaDataset(settings.SCHEMAGUIDED_PATH, split=datasetconfig['split'])

    if dataset == 'opendialkg':
        dataset_reader = opendialkg_reader.OpendialkgDataset(settings.OPENDIALKG_PATH, split=datasetconfig['split'])

    if dataset == 'flodial':
        dataset_reader = flodial_reader.FlodialDataset(settings.FLODIAL_PATH, split=datasetconfig['split'])

    if dataset == 'toxichat':
        dataset_reader = toxichat_reader.ToxichatDataset(settings.TOXICHAT_PATH, split=datasetconfig['split'])

    if dataset == 'bad':
        dataset_reader = bad_reader.BadDataset(settings.BAD_PATH, split=datasetconfig['split'])

    if dataset == 'buildbreakfix':
        dataset_reader = buildbreakfix_reader.BuildBreakFixDataset(settings.BUILDBREAKFIX_PATH,
                                                                   split=datasetconfig['split'])

    if dataset == 'saferdialogues':
        dataset_reader = saferdialogues_reader.SaferDialoguesDataset(settings.SAFERDIALOGUES_PATH,
                                                                     split=datasetconfig['split'])

    if dataset == 'gensf':
        dataset_reader = gensf_reader.GensfDataset(settings.GENSF_PATH, split=datasetconfig['split'],
                                                   domain=datasetconfig.get('domain', 'Events_1'))

    if dataset == 'kvret':
        dataset_reader = kvret_reader.KvretDataset(settings.KVRET_PATH, split=datasetconfig['split'])

    if dataset == 'wozdst':
        dataset_reader = wozdst_reader.WozDstDataset(settings.WOZDST_PATH, split=datasetconfig['split'])

    if dataset == 'taskmasterdst':
        dataset_reader = taskmasterdst_reader.TaskMaster(settings.TASKMASTERDST_PATH, split=datasetconfig['split'])

    if dataset == 'msre2edst':
        dataset_reader = msre2edst_reader.Msre2eDst(settings.ME2EDST_PATH, split=datasetconfig['split'])

    if dataset == 'multiwozdst':
        dataset_reader = multiwozdst_reader.MultiwozDstDataset(settings.MULTIWOZDST_PATH, split=datasetconfig['split'])

    dataset_reader.split = datasetconfig['split']

    return dataset_reader
