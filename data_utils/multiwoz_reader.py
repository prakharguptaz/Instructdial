from data_utils.data_reader import Dataset
from datasets import load_dataset


class MultiwozDataset(Dataset):
    def __init__(self, split='train'):
        self.examples = []

        # ignore download checksum, see: https://github.com/huggingface/datasets/issues/1876
        dataset = load_dataset('multi_woz_v22', ignore_verifications=True)
        data = dataset[split]

        for idx, dp in enumerate(data):
            turns = dp['turns']
            lines = []
            for utterance, speaker, acts, frames in zip(turns['utterance'],
                                                        turns['speaker'],
                                                        turns['dialogue_acts'],
                                                        turns['frames']):

                act = ''
                if len(acts['dialog_act']['act_type']) > 0:
                    act = acts['dialog_act']['act_type'][0]

                state = []
                if frames['state']:
                    state = [s['active_intent'] for s in frames['state']]

                slots = utterance
                span_info = acts['span_info']
                for slot_name, span_start, span_end in zip(reversed(span_info['act_slot_name']),
                                                           reversed(span_info['span_start']),
                                                           reversed(span_info['span_end'])):
                    slots = slots[:span_start] + 'SLOT_' + slot_name + slots[span_end:]
                slots = ' '.join(
                    [s.replace('SLOT_', '') if s.startswith('SLOT_') else 'O' for s in slots.split()])

                lines.append(
                    {
                        'utterance': utterance,
                        'speaker': speaker,
                        'slots': slots,
                        'act': act,
                        'state': state
                    }
                )
            self.examples.append({
                'dialogue_id': dp['dialogue_id'],
                'services': dp['services'],
                'turns': lines
            })
        print()
