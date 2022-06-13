import os
from data_utils.data_reader import Dataset
import csv
import logging
import ast
import pdb
import random

TARGET_GROUPS = ["celebrity/personality", "individual/redditor", "comment author", "black folks", "asian folks",
                 "latino/latina folks", "native american/first nation folks", "people of color (POC)", "women", "men",
                 "LGBTQ folks", "feminists", "christian folks", "muslim folks", "jewish folks", "arabic folks",
                 "religious folks/theists", "atheists", "old folks/seniors", "veterans", "young folks/kids/babies",
                 "overweight/fat folks", "shorts folks", "pregnant folks",
                 "folks with physical disability/illness/disorder", "folks with mental disability/illness/disorder",
                 "harassment/assault victims", "climate disaster victims", "mass shooting victims", "terrorism victims",
                 "leftists", "rightists", "centrists", "liberals", "conservatives", "independents/libertarians",
                 "anarchists", "socialists", "communists", "immigrants", "people from a region", "republicans",
                 "democrats", "poor folks", "Not in the list"]

TARGET_GROUPS_TO_ID = {group: i for i, group in enumerate(TARGET_GROUPS)}
OFFENSIVE_SUBREDDITS = {"AskThe_Donald", "Braincels", "MensRights", "MGTOW", "TwoXChromosomes", "atheism",
                        "Libertarian", "unpopularopinion", "islam", "lgbt"}


class ToxichatDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.idx = 0
        self.examples = []

        conversations_data, header = get_conversation_data_from_OC_S_file(
            os.path.join(data_path, 'OC_S_{}.csv'.format(split)))

        for item in conversations_data:
            for comment in item.utterance_data:
                toxic = False
                if comment['off'] == 1:
                    toxic = True
                self.examples.append({
                    "context": [],
                    "response": comment['comment'][1:],
                    "toxic": toxic
                })


def process_stance_annotation(stance_annotation):
    return int(stance_annotation) if stance_annotation else None


def process_target_annotation(target_annotation):
    return set() if target_annotation == "set()" else ast.literal_eval(target_annotation)


def normalize_stance_label(stance_label):
    stance_label_mapping = {0: 0, 1: 1, -1: 2, None: None}
    return stance_label_mapping[stance_label]


class Conversation_Data(object):
    def extract_data_from_rows(self, rows, flat_OC_S=False):
        if type(rows) == tuple:
            if len(rows) == 2:
                if type(rows[0]) == tuple:
                    # OC_S pairwise stance instance
                    self.data_source = "OC_S_pairwise_stance"
                    # The rows here is a tuple of (post, tuple of 5 labels with offend label at position 0)
                    (u_toward, u_response), stance_label = rows
                    u_data = {"id": 1, "stance": stance_label, "comment": f"{u_toward} [SEP] {u_response}"}
                    self.utterance_data = [u_data]

                    # initialize everything else to None
                    self.subreddit, self.sample_type, self.thread_id, self.last_off_score = None, None, None, None
                else:
                    # SBF data
                    self.data_source = "SBF"
                    # The rows here is a tuple of (post, tuple of 5 labels with offend label at position 0)
                    post, labels = rows
                    off_label = labels[0]
                    u_data = {"id": 1, "off": off_label, "comment": post}
                    self.utterance_data = [u_data]

                    # initialize everything else to None
                    self.subreddit, self.sample_type, self.thread_id, self.last_off_score = None, None, None, None
            elif len(rows) == 4:
                # OC_S data
                self.data_source = "OC_S_flat"
                # The rows here is a tuple of (utterance, off_label, u_id, conv_signature)
                utterance, off_label, id, conv_signature = rows
                u_data = {"id": id, "off": off_label, "comment": utterance}
                self.utterance_data = [u_data]

                # initialize the data signature to conv_signature
                self.subreddit, self.sample_type, self.thread_id, self.last_off_score = conv_signature
            else:
                logging.error(
                    f"Illegal row size = {len(rows)} given input to Conversation Data. Expected len to be 2 or 4. Terminating!")
                exit(1)
            self.dgpt_resp_data = None
            self.gpt3_resp_data = None
        else:
            # OC_S data
            self.data_source = "OC_S"
            response_rows = rows[-2:]
            utterance_rows = rows[1:-2]
            conversation_info_row = rows[0]
            # print(conversation_info_row)
            # Extract data from Utterance Annotations
            utterance_data = list()
            for i, utterance_row in enumerate(utterance_rows):
                u_id = i + 1
                current_u_data = dict()
                current_u_data["id"] = u_id
                current_u_data["comment"] = utterance_row[0]
                current_u_data["off"] = int(utterance_row[1])
                current_u_data["targets"] = process_target_annotation(utterance_row[2])
                for j in range(1, u_id):
                    current_u_data[f"{j}stance"] = process_stance_annotation(utterance_row[2 + j])
                utterance_data.append(current_u_data)
            # Extract data from Response Annotations
            dgpt_resp_data = {"resp_type": "dgpt", "id": "dgpt"}
            gpt3_resp_data = {"resp_type": "gpt3", "id": "gpt3"}
            for response_row in response_rows:
                comment = response_row[0]
                if comment.startswith("DGPT_response:"):
                    comment = comment.replace("DGPT_response:", "").strip()
                    current_resp_data = dgpt_resp_data
                elif comment.startswith("GPT3_response:"):
                    comment = comment.replace("GPT3_response:", "").strip()
                    current_resp_data = gpt3_resp_data
                current_resp_data["comment"] = comment
                current_resp_data["off"] = int(response_row[1])
                current_resp_data["targets"] = process_target_annotation(response_row[2])
                for j in range(1, 4):
                    current_resp_data[f"{j}stance"] = process_stance_annotation(response_row[2 + j])
                current_resp_data["coherence"] = response_row[-1]
            # print_list(utterance_data)
            # print(dgpt_resp_data)
            # print(gpt3_resp_data)
            # Save everything in the correct objects

            # Remove empty elements from conversation row
            conversation_info_row = [e for e in conversation_info_row if e]

            if len(conversation_info_row) == 5:
                self.subset, self.thread_id, self.sample_type, self.subreddit, self.last_off_score = conversation_info_row
            else:
                logging.warn(f"current conv_data is in old format {conversation_info_row}")
                self.thread_id, self.sample_type, self.subreddit, self.last_off_score = conversation_info_row[:4]
            self.utterance_data = utterance_data
            self.dgpt_resp_data = dgpt_resp_data
            self.gpt3_resp_data = gpt3_resp_data

    """We will store all the data and annotations related to a conversation in the objects of this class"""

    def __init__(self, conversation_rows, flat_OC_S=False):
        super(Conversation_Data, self).__init__()
        self.conversation_rows = conversation_rows
        # We will extract the conversation and annotation elements from these rows
        self.extract_data_from_rows(self.conversation_rows, flat_OC_S)

    def get_processed_utterance_from_id(self, u_id):
        # Get the offensive label for the given u_id
        if u_id == "dgpt":
            return self.dgpt_resp_data["comment"]
        if u_id == "gpt3":
            return self.gpt3_resp_data["comment"]
        assert type(u_id) == int, f"Given utterance id = {u_id} is neither string nor int"
        # u_id will be 1, 2 and 3. Therefore, we will decrease 1 from u_id
        u_id = u_id - 1
        if u_id < len(self.utterance_data):
            return self.utterance_data[u_id]["comment"][
                   2:].strip()  # removing the first 2 characters which are the emoji
        return None

    def get_off_label(self, u_id):
        # Get the offensive label for the given u_id
        if u_id == "dgpt":
            return self.dgpt_resp_data["off"]
        if u_id == "gpt3":
            return self.gpt3_resp_data["off"]
        assert type(u_id) == int, f"Given utterance id = {u_id} is neither string nor int"
        # u_id will be 1, 2 and 3. Therefore, we will decrease 1 from u_id
        u_id = u_id - 1
        if u_id < len(self.utterance_data):
            return self.utterance_data[u_id]["off"]
        return None

    def get_off_prediction(self, u_id):
        # Get the offensive prediction for the given u_id
        if u_id == "dgpt":
            return self.dgpt_resp_data["off_prediction"]
        if u_id == "gpt3":
            return self.gpt3_resp_data["off_prediction"]
        assert type(u_id) == int, f"Given utterance id = {u_id} is neither string nor int"
        # u_id will be 1, 2 and 3. Therefore, we will decrease 1 from u_id
        u_id = u_id - 1
        if u_id < len(self.utterance_data):
            return self.utterance_data[u_id]["off_prediction"]
        return None

    def get_off_prediction_score(self, u_id):
        # Get the offensive prediction_score for the given u_id
        if u_id == "dgpt":
            return self.dgpt_resp_data["off_prediction_score"]
        if u_id == "gpt3":
            return self.gpt3_resp_data["off_prediction_score"]
        assert type(u_id) == int, f"Given utterance id = {u_id} is neither string nor int"
        # u_id will be 1, 2 and 3. Therefore, we will decrease 1 from u_id
        u_id = u_id - 1
        if u_id < len(self.utterance_data):
            return self.utterance_data[u_id]["off_prediction_score"]
        return None

    def get_off_targets(self, u_id):
        # Get the offensive targets for the given u_id
        if u_id == "dgpt":
            return self.dgpt_resp_data["targets"]
        if u_id == "gpt3":
            return self.gpt3_resp_data["targets"]
        assert type(u_id) == int, f"Given utterance id = {u_id} is neither string nor int"
        # u_id will be 1, 2 and 3. Therefore, we will decrease 1 from u_id
        u_id = u_id - 1
        if u_id < len(self.utterance_data):
            return self.utterance_data[u_id]["targets"]
        return None

    def get_stance_label(self, from_id, to_id):
        # Get the stance label for the given pair of ids
        if from_id == "dgpt":
            return self.dgpt_resp_data[f"{to_id}stance"]
        if from_id == "gpt3":
            return self.gpt3_resp_data[f"{to_id}stance"]
        from_id -= 1
        if from_id < len(self.utterance_data):
            return self.utterance_data[from_id][f"{to_id}stance"]
        return None

    def set_stance_prediction_and_score(self, from_id, to_id, prediction, score):
        # Get the stance label for the given pair of ids
        u_data = None
        if from_id == "dgpt":
            u_data = self.dgpt_resp_data
        elif from_id == "gpt3":
            u_data = self.gpt3_resp_data
        else:
            assert type(from_id) == int, pdb.set_trace()
            from_id -= 1
            assert from_id < len(self.utterance_data)
            u_data = self.utterance_data[from_id]
        u_data.setdefault(f"{to_id}stance_prediction", list())
        u_data.setdefault(f"{to_id}stance_prediction_score", list())
        u_data[f"{to_id}stance_prediction"].append(prediction)
        u_data[f"{to_id}stance_prediction_score"].append(score)
        return None

    def get_stance_predictions_scores_and_labels_for_u_id(self, from_id, adjacent_only=False):
        u_data = None
        n_previous = 0
        if from_id == "dgpt":
            u_data = self.dgpt_resp_data
            n_previous = len(self.utterance_data) + 1
        elif from_id == "gpt3":
            u_data = self.gpt3_resp_data
            n_previous = len(self.utterance_data) + 1
        else:
            assert type(from_id) == int, pdb.set_trace()
            from_id -= 1
            if from_id >= len(self.utterance_data):
                return None, None, None
            u_data = self.utterance_data[from_id]
            n_previous = from_id + 1
        predictions = list()
        scores = list()
        labels = list()
        for j in range(1, n_previous):
            if f"{j}stance_prediction" not in u_data:
                continue
            if adjacent_only and j != (n_previous - 1):
                continue
            predictions.append(u_data[f"{j}stance_prediction"])
            scores.append(u_data[f"{j}stance_prediction_score"])
            labels.append(normalize_stance_label(u_data[f"{j}stance"]))
        return predictions, scores, labels

    def get_stance_prediction(self, from_id, to_id):
        # Get the stance prediction for the given pair of ids
        if from_id == "dgpt":
            if f"{to_id}stance_prediction" not in self.dgpt_resp_data:
                return None
            return self.dgpt_resp_data[f"{to_id}stance_prediction"]
        if from_id == "gpt3":
            if f"{to_id}stance_prediction" not in self.gpt3_resp_data:
                return None
            return self.gpt3_resp_data[f"{to_id}stance_prediction"]
        from_id -= 1
        if from_id < len(self.utterance_data):
            if f"{to_id}stance_prediction" not in self.utterance_data[from_id]:
                return None
            return self.utterance_data[from_id][f"{to_id}stance_prediction"]
        return None

    def get_stance_prediction_score(self, from_id, to_id):
        # Get the stance prediction score for the given pair of ids
        if from_id == "dgpt":
            if f"{to_id}stance_prediction_score" not in self.dgpt_resp_data:
                return None
            return self.dgpt_resp_data[f"{to_id}stance_prediction_score"]
        if from_id == "gpt3":
            if f"{to_id}stance_prediction_score" not in self.gpt3_resp_data:
                return None
            return self.gpt3_resp_data[f"{to_id}stance_prediction_score"]
        from_id -= 1
        if from_id < len(self.utterance_data):
            if f"{to_id}stance_prediction_score" not in self.utterance_data[from_id]:
                return None
            return self.utterance_data[from_id][f"{to_id}stance_prediction_score"]
        return None

    def log_offensive_prediction(self, u_id=None):
        logging.info((self.subset, self.subreddit, self.sample_type, self.thread_id, self.last_off_score))
        for u_data in self.utterance_data:
            prefix_string = "** " if u_data["id"] == u_id else ""
            logging.info(
                f"{prefix_string}{u_data['off_prediction']}\t{u_data['off_prediction_score']:.4f}|{u_data['off']}\t{u_data['comment']}")
        prefix_string = "** " if "dgpt" == u_id else ""
        logging.info(
            f"{prefix_string}DGPT - {self.dgpt_resp_data['off_prediction']}\t{self.dgpt_resp_data['off_prediction_score']:.4f}|{self.dgpt_resp_data['off']}\t{self.dgpt_resp_data['comment']}")
        prefix_string = "** " if "gpt3" == u_id else ""
        logging.info(
            f"{prefix_string}GPT3 - {self.gpt3_resp_data['off_prediction']}\t{self.gpt3_resp_data['off_prediction_score']:.4f}|{self.gpt3_resp_data['off']}\t{self.gpt3_resp_data['comment']}\n")

    def log_stance_prediction(self, from_id, to_id):
        analysis_csv_rows = list()
        logging.info((self.subset, self.subreddit, self.sample_type, self.thread_id, self.last_off_score))
        analysis_csv_rows.append([(self.subset, self.subreddit, self.sample_type, self.thread_id, self.last_off_score)])
        for u_data in self.utterance_data:
            u_id = u_data["id"]
            if u_id < 2:
                # first comment
                logging.info(f"{u_data['comment']}")
                analysis_csv_rows.append(["", u_data['comment']])
                continue
            stance_predictions_and_labels_str = list()
            for j in range(1, u_id):
                prefix_string = "** " if u_data["id"] == from_id and j == to_id else ""
                if f"{j}stance_prediction" in u_data:
                    score = u_data[f'{j}stance_prediction_score']
                    stance_predictions_and_labels_str.append(
                        f"{prefix_string}{u_data[f'{j}stance_prediction']}|{score[0]:.3f},{score[1]:.3f},{score[2]:.3f}|{normalize_stance_label(u_data[f'{j}stance'])}")
            stance_predictions_and_labels_str = '||'.join(stance_predictions_and_labels_str)
            analysis_csv_rows.append([stance_predictions_and_labels_str, u_data['comment']])
            logging.info(f"{stance_predictions_and_labels_str}\t{u_data['comment']}")
        n_utterances = len(self.utterance_data)
        for resp_data in [self.dgpt_resp_data, self.gpt3_resp_data]:
            stance_predictions_and_labels_str = list()
            for j in range(1, n_utterances + 1):
                prefix_string = "** " if resp_data["id"] == from_id and j == to_id else ""
                if f"{j}stance_prediction" in resp_data:
                    score = resp_data[f'{j}stance_prediction_score']
                    stance_predictions_and_labels_str.append(
                        f"{prefix_string}{resp_data[f'{j}stance_prediction']}|{score[0]:.3f},{score[1]:.3f},{score[2]:.3f}|{normalize_stance_label(resp_data[f'{j}stance'])}")
            stance_predictions_and_labels_str = '||'.join(stance_predictions_and_labels_str)
            analysis_csv_rows.append(
                [f"{resp_data['id']} response - {stance_predictions_and_labels_str}", resp_data['comment']])
            logging.info(f"{resp_data['id']} response - {stance_predictions_and_labels_str}\t{resp_data['comment']}")
        return analysis_csv_rows

    def print_conv(self):
        # Print the conversation data
        print(self.subreddit, self.sample_type, self.thread_id, self.last_off_score)
        print_list(self.utterance_data)
        print(self.dgpt_resp_data)
        print(self.gpt3_resp_data)
        print()


def print_list(l, K=None):
    # If K is given then only print first K
    for i, e in enumerate(l):
        if i == K:
            break
        print(e)
    print()


def check_if_list_is_empty_or_empty_elements(l):
    if len(l) == 0:
        return True
    modified_l = [a for a in l if a != ""]
    if len(modified_l) == 0:
        return True
    return False


def get_conversation_data_from_SBF_instances(SBF_instances):
    return [Conversation_Data(sbf_instance) for sbf_instance in SBF_instances]


def get_OC_S_flat_conversation_data_from_OC_S_comment_data(comment_data, conv_signature):
    # conv_signature = tuple of (subreddit, sample_type, thread_id, last_off_score)
    comment = comment_data["comment"]
    off_label = comment_data["off"]
    id = comment_data["id"]
    assert type(comment) == str
    assert type(off_label) == int
    # create a new flat conversation_data
    if type(id) == int:
        # Remove the first emoji when creating flat OC_S data
        comment = comment[2:]
    flat_conversation_data = Conversation_Data((comment, off_label, id, conv_signature))
    return flat_conversation_data


def load_from_tsv_to_list_of_list(tsv_file, delimiter="\t", header_present=False):
    # Load the TSV into a list of list
    all_rows = list()
    with open(tsv_file, "r") as tsv_in:
        reader = csv.reader(tsv_in, delimiter=delimiter)
        if header_present:
            header = next(reader)
        all_rows = [row for row in reader]
    if header_present:
        return all_rows, header
    return all_rows


def get_conversation_data_from_OC_S_file(OC_S_file, flat_OC_S=False):
    dataset_rows, header = load_from_tsv_to_list_of_list(OC_S_file, delimiter=",", header_present=True)
    # header = ['utterance', 'uOff', 'uOffTarget', 'u1stance', 'u2stance', 'u3stance', 'resp_coherence']
    logging.info(f"DATASET ROWS = {len(dataset_rows)}")
    accumulated_rows = list()
    conversations_data = list()
    for row in dataset_rows:
        if check_if_list_is_empty_or_empty_elements(row):
            # create conversation
            conversations_data.append(Conversation_Data(accumulated_rows))
            # reset row accumulator
            accumulated_rows = list()
        else:
            accumulated_rows.append(row)
    if len(accumulated_rows) > 0:
        # Create last conversation
        conversations_data.append(Conversation_Data(accumulated_rows))
    logging.info(f"Conversation Data = {len(conversations_data)}")
    if flat_OC_S:
        logging.info(f"Flattening OC_S data..")
        flat_conversations_data = list()
        for conversation_data in conversations_data:
            # Get the conversation signature from the original object
            signature = (conversation_data.subreddit, conversation_data.sample_type, conversation_data.thread_id,
                         conversation_data.last_off_score)
            # For each pairwise stance variable create a new conversation

            # For each utterance and response create a new conversation_data similar to "SBF"
            for utterance_data in conversation_data.utterance_data:
                flat_conversations_data.append(
                    get_OC_S_flat_conversation_data_from_OC_S_comment_data(utterance_data, signature))
            # Add one flat conversation_data for dgpt response and gpt3 response
            flat_conversations_data.append(
                get_OC_S_flat_conversation_data_from_OC_S_comment_data(conversation_data.dgpt_resp_data, signature))
            flat_conversations_data.append(
                get_OC_S_flat_conversation_data_from_OC_S_comment_data(conversation_data.gpt3_resp_data, signature))
        logging.info(f"Final flat conversations = {len(flat_conversations_data)}")
        conversations_data = flat_conversations_data

    return conversations_data, header


def get_pairwise_stance_conversation_data_from_OC_S_file(OC_S_file):
    dataset_rows, header = load_from_tsv_to_list_of_list(OC_S_file, delimiter=",", header_present=True)
    # header = ['utterance', 'uOff', 'uOffTarget', 'u1stance', 'u2stance', 'u3stance', 'resp_coherence']
    logging.info(f"DATASET ROWS = {len(dataset_rows)}")
    accumulated_rows = list()
    conversations_data = list()
    for row in dataset_rows:
        if check_if_list_is_empty_or_empty_elements(row):
            # create conversation
            conversations_data.append(Conversation_Data(accumulated_rows))
            # reset row accumulator
            accumulated_rows = list()
        else:
            accumulated_rows.append(row)
    if len(accumulated_rows) > 0:
        # Create last conversation
        conversations_data.append(Conversation_Data(accumulated_rows))
    logging.info(f"Conversation Data = {len(conversations_data)}")

    # Now we convert the full conversations data into pairwise stance conversation data
    new_pairwise_stance_convs = list()
    for conv in conversations_data:
        stance_labels = list()
        stance_u_pairs = list()
        for current_u_data in conv.utterance_data:
            u_id = current_u_data["id"]
            u = current_u_data["comment"][2:].strip()
            # Ignore the first utterance
            if u_id < 2:
                continue
            # Find stance labels and u_id pairs for previous utterances and save them in lists
            for i in range(1, u_id):
                u_id_pair = (i, u_id)
                u_pair = (conv.utterance_data[i - 1]["comment"][2:].strip(), u)
                stance_label = current_u_data[f"{i}stance"]
                stance_labels.append(normalize_stance_label(stance_label))
                stance_u_pairs.append(u_pair)
        # Create stance labels and u_id pairs for DGPT and GPT3 responses
        n_thread_utterances = len(conv.utterance_data)
        u_id = n_thread_utterances + 1
        for i in range(1, n_thread_utterances + 1):
            u_id_pair = (i, u_id)
            u_toward = conv.utterance_data[i - 1]["comment"][2:].strip()
            u_dgpt = conv.dgpt_resp_data["comment"].strip()
            u_gpt3 = conv.gpt3_resp_data["comment"].strip()
            dgpt_stance_label = conv.dgpt_resp_data[f"{i}stance"]
            gpt3_stance_label = conv.gpt3_resp_data[f"{i}stance"]
            stance_labels.append(normalize_stance_label(dgpt_stance_label))
            stance_u_pairs.append((u_toward, u_dgpt))
            stance_labels.append(normalize_stance_label(gpt3_stance_label))
            stance_u_pairs.append((u_toward, u_gpt3))
        # Create conversation data from stance_labels and stance_u_pairs
        for stance_u_pair, stance_label in zip(stance_u_pairs, stance_labels):
            new_pairwise_stance_convs.append(Conversation_Data((stance_u_pair, stance_label)))

    return new_pairwise_stance_convs, header


def get_save_lists_from_conv_data(conversation_data):
    save_rows = list()
    for conv in conversation_data:
        save_rows.extend(conv.conversation_rows)
        save_rows.append([])
    return save_rows


def normalize_targets(targets):
    if len(targets) == 0:
        return list()
    return [TARGET_GROUPS_TO_ID[group] for group in targets]


def create_instances_from_convs(conversations_data, stance=False):
    instances = list()
    for conv in conversations_data:
        if conv.data_source == "OC_S":
            subreddit, sample_type, thread_id = conv.subreddit, conv.sample_type, conv.thread_id
            # Extract conversational utterances and off_labels
            utterances = list()
            off_labels = list()
            off_targets = list()

            for current_u_data in conv.utterance_data:
                # u_id = current_u_data["id"]
                utterances.append(current_u_data["comment"][2:])
                off_labels.append(current_u_data["off"])
                off_targets.append(normalize_targets(current_u_data["targets"]))
            # Create 2 instances from each conv_data. 1 for DGPT response and 1 for GPT3 response
            dgpt_utterances = utterances + [conv.dgpt_resp_data["comment"]]
            dgpt_off_labels = off_labels + [conv.dgpt_resp_data["off"]]
            dgpt_off_targets = off_targets + [normalize_targets(conv.dgpt_resp_data["targets"])]
            gpt3_utterances = utterances + [conv.gpt3_resp_data["comment"]]
            gpt3_off_labels = off_labels + [conv.gpt3_resp_data["off"]]
            gpt3_off_targets = off_targets + [normalize_targets(conv.gpt3_resp_data["targets"])]

            if not stance:
                # No stance required. Directly update the instance list
                instances.append({"source": conv.data_source, "subreddit": subreddit, "sample_type": sample_type,
                                  "thread_id": thread_id, "resp_type": "dgpt", "utterances": dgpt_utterances,
                                  "off_labels": dgpt_off_labels, "off_targets": dgpt_off_targets})
                instances.append({"source": conv.data_source, "subreddit": subreddit, "sample_type": sample_type,
                                  "thread_id": thread_id, "resp_type": "gpt3", "utterances": gpt3_utterances,
                                  "off_labels": gpt3_off_labels, "off_targets": gpt3_off_targets})
            else:
                # If stance flag is given then collect stance labels and stance u_id pairs
                stance_labels = list()
                stance_u_id_pairs = list()
                for current_u_data in conv.utterance_data:
                    u_id = current_u_data["id"]
                    # Ignore the first utterance
                    if u_id < 2:
                        continue
                    # Find stance labels and u_id pairs for previous utterances and save them in lists
                    for i in range(1, u_id):
                        u_id_pair = (i, u_id)
                        stance_label = current_u_data[f"{i}stance"]
                        stance_labels.append(normalize_stance_label(stance_label))
                        stance_u_id_pairs.append(u_id_pair)
                # Create stance labels and u_id pairs for DGPT and GPT3 responses
                n_thread_utterances = len(conv.utterance_data)
                dgpt_stance_labels = list()
                dgpt_stance_u_id_pairs = list()
                gpt3_stance_labels = list()
                gpt3_stance_u_id_pairs = list()
                u_id = n_thread_utterances + 1
                for i in range(1, n_thread_utterances + 1):
                    u_id_pair = (i, u_id)
                    dgpt_stance_label = conv.dgpt_resp_data[f"{i}stance"]
                    gpt3_stance_label = conv.gpt3_resp_data[f"{i}stance"]
                    dgpt_stance_labels.append(normalize_stance_label(dgpt_stance_label))
                    dgpt_stance_u_id_pairs.append(u_id_pair)
                    gpt3_stance_labels.append(normalize_stance_label(gpt3_stance_label))
                    gpt3_stance_u_id_pairs.append(u_id_pair)
                # Add the utterance stance with dgpt and gpt3 stance
                dgpt_stance_labels = stance_labels + dgpt_stance_labels
                dgpt_stance_u_id_pairs = stance_u_id_pairs + dgpt_stance_u_id_pairs
                gpt3_stance_labels = stance_labels + gpt3_stance_labels
                gpt3_stance_u_id_pairs = stance_u_id_pairs + gpt3_stance_u_id_pairs

                # Add everything to the instance list
                instances.append({"source": conv.data_source, "subreddit": subreddit, "sample_type": sample_type,
                                  "thread_id": thread_id, "resp_type": "dgpt", "utterances": dgpt_utterances,
                                  "off_labels": dgpt_off_labels, "off_targets": dgpt_off_targets,
                                  "stance_labels": dgpt_stance_labels, "stance_u_id_pairs": dgpt_stance_u_id_pairs})
                instances.append({"source": conv.data_source, "subreddit": subreddit, "sample_type": sample_type,
                                  "thread_id": thread_id, "resp_type": "gpt3", "utterances": gpt3_utterances,
                                  "off_labels": gpt3_off_labels, "off_targets": gpt3_off_targets,
                                  "stance_labels": gpt3_stance_labels, "stance_u_id_pairs": gpt3_stance_u_id_pairs})

        elif conv.data_source == "OC_S_flat":
            utterances = [conv.utterance_data[0]["comment"]]
            off_labels = [conv.utterance_data[0]["off"]]
            ids = [conv.utterance_data[0]["id"]]
            instances.append({"source": conv.data_source, "subreddit": conv.subreddit, "sample_type": conv.sample_type,
                              "thread_id": conv.thread_id, "resp_type": None, "id": ids, "utterances": utterances,
                              "off_labels": off_labels})
        elif conv.data_source == "SBF":
            utterances = [conv.utterance_data[0]["comment"]]
            off_labels = [conv.utterance_data[0]["off"]]
            instances.append({"source": conv.data_source, "subreddit": None, "sample_type": None, "thread_id": None,
                              "resp_type": None, "utterances": utterances, "off_labels": off_labels})
        elif conv.data_source == "OC_S_pairwise_stance":
            # Create instances for pairwise stance
            utterance_pairs = [conv.utterance_data[0]["comment"]]
            stance_labels = [conv.utterance_data[0]["stance"]]
            instances.append({"source": conv.data_source, "subreddit": None, "sample_type": None, "thread_id": None,
                              "resp_type": None, "utterances": utterance_pairs, "stance_labels": stance_labels})
        else:
            logging.error(f"Unrecognized data from source = {conv.data_source}")
            exit()
    return instances


class OC_S_BERT_Dataset(Dataset):
    """OC_S_BERT_Dataset stores the OC_S instances. It takes list of Conversation_Data.
        It transforms the input list of Conversation_Data into list of dictionary instances"""

    def __init__(self, conversations_data, stance=False):
        super(OC_S_BERT_Dataset, self).__init__()
        self.instances = create_instances_from_convs(conversations_data, stance)
        self.nsamples = len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return self.nsamples


def log_TP_FP_FN_TN_from_conv_off_predictions(predictions, scores, labels, convs, K=10):
    # Given binary predictions, gold labels and instances we will find instances that are TP, FP, FN and TN
    # Then we will log a sample of K instances from each category for verification
    categories = ["TP", "FP", "FN", "TN"]
    category_explanations = {"TP": "prediction = 1 and label = 1", "FP": "prediction = 1 and label = 0",
                             "FN": "prediction = 0 and label = 1", "TN": "prediction = 0 and label = 0"}
    category_instances = {category: list() for category in categories}
    for conv_prediction, conv_scores, conv_label, conv in zip(predictions, scores, labels, convs):
        for index, (prediction, score, label) in enumerate(zip(conv_prediction, conv_scores, conv_label)):
            if prediction == 1 and label == 1:
                # TP
                category_instances["TP"].append((index, conv_prediction, conv_scores, conv_label, conv))
            elif prediction == 1 and label == 0:
                # FP
                category_instances["FP"].append((index, conv_prediction, conv_scores, conv_label, conv))
            elif prediction == 0 and label == 1:
                # FN
                category_instances["FN"].append((index, conv_prediction, conv_scores, conv_label, conv))
            elif prediction == 0 and label == 0:
                # TN
                category_instances["TN"].append((index, conv_prediction, conv_scores, conv_label, conv))
            else:
                # Incorrect prediction or label
                logging.error(f"Incorrect prediction({prediction}) or label({label})")
                exit(1)
    # Log a sample form each category
    for category in categories:
        if len(category_instances[category]) <= K:
            sample_size = len(category_instances[category])
        else:
            sample_size = K
        logging.info(
            f"{category}:{category_explanations[category]}:A sample of {sample_size}/{len(category_instances[category])} instances:")
        category_sample = random.sample(category_instances[category], sample_size)
        log_list(category_sample)


def log_list(l, K=None):
    # If K is given then only log first K
    for i, e in enumerate(l):
        if i == K:
            break
        logging.info(e)
    logging.info("")


def create_offensive_instances_from_convs(conversations_data, flat=False):
    instances = list()
    for conv in conversations_data:
        if conv.data_source == "OC_S":
            subreddit, sample_type, thread_id = conv.subreddit, conv.sample_type, conv.thread_id
            # Extract conversational utterances and off_labels
            u_ids = list()
            utterances = list()
            off_labels = list()
            off_targets = list()

            for i, current_u_data in enumerate(conv.utterance_data):
                if i == 0:
                    # First comment data is the post
                    comment = current_u_data["comment"][2:].strip()
                    comment = f"subreddit = {conv.subreddit} {comment}"
                    utterances.append(comment)
                else:
                    utterances.append(current_u_data["comment"][2:])
                u_ids.append(current_u_data["id"])
                off_labels.append(current_u_data["off"])
                off_targets.append(normalize_targets(current_u_data["targets"]))
            # Create 2 instances from each conv_data. 1 for DGPT response and 1 for GPT3 response
            dgpt_utterances = utterances + [conv.dgpt_resp_data["comment"]]
            dgpt_off_labels = off_labels + [conv.dgpt_resp_data["off"]]
            dgpt_off_targets = off_targets + [normalize_targets(conv.dgpt_resp_data["targets"])]
            gpt3_utterances = utterances + [conv.gpt3_resp_data["comment"]]
            gpt3_off_labels = off_labels + [conv.gpt3_resp_data["off"]]
            gpt3_off_targets = off_targets + [normalize_targets(conv.gpt3_resp_data["targets"])]

            if flat:
                # Create single utterance instances instead of a sequence for all utterances in the thread
                for u_id, u, off_label, off_target in zip(u_ids, utterances, off_labels, off_targets):
                    instances.append(
                        {"conv": conv, "resp_type": None, "id": u_id, "utterances": u, "off_labels": off_label,
                         "off_targets": off_target})
                # Create single utterance instances for DGPT and GPT3 responses
                instances.append(
                    {"conv": conv, "resp_type": "dgpt", "id": "dgpt", "utterances": conv.dgpt_resp_data["comment"],
                     "off_labels": conv.dgpt_resp_data["off"],
                     "off_targets": normalize_targets(conv.dgpt_resp_data["targets"])})
                instances.append(
                    {"conv": conv, "resp_type": "gpt3", "id": "gpt3", "utterances": conv.gpt3_resp_data["comment"],
                     "off_labels": conv.gpt3_resp_data["off"],
                     "off_targets": normalize_targets(conv.gpt3_resp_data["targets"])})

            else:
                # No stance required. Directly update the instance list
                instances.append(
                    {"conv": conv, "resp_type": "dgpt", "utterances": dgpt_utterances, "off_labels": dgpt_off_labels,
                     "off_targets": dgpt_off_targets})
                instances.append(
                    {"conv": conv, "resp_type": "gpt3", "utterances": gpt3_utterances, "off_labels": gpt3_off_labels,
                     "off_targets": gpt3_off_targets})

        elif conv.data_source == "OC_S_flat":
            utterances = [conv.utterance_data[0]["comment"]]
            off_labels = [conv.utterance_data[0]["off"]]
            ids = [conv.utterance_data[0]["id"]]
            instances.append(
                {"conv": conv, "resp_type": None, "id": ids, "utterances": utterances, "off_labels": off_labels})
        elif conv.data_source == "SBF":
            utterances = [conv.utterance_data[0]["comment"]]
            off_labels = [conv.utterance_data[0]["off"]]
            instances.append({"conv": conv, "resp_type": None, "utterances": utterances, "off_labels": off_labels})
        else:
            logging.error(f"Unrecognized data from source = {conv.data_source}")
            exit()
    return instances


class OC_S_offensive_Dataset(Dataset):
    """OC_S_offensive_Dataset stores the OC_S_post_thread instances. It takes list of Conversation_Data.
        It transforms the input list of Conversation_Data into list of dictionary instances"""

    def __init__(self, conversations_data, flat_only=False):
        super(OC_S_offensive_Dataset, self).__init__()
        self.instances = create_offensive_instances_from_convs(conversations_data, flat_only)
        self.nsamples = len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return self.nsamples


def log_TP_FP_FN_TN_convs_from_off_predictions(id_to_conv, K=5):
    categories = ["TP", "FP", "FN", "TN"]
    category_explanations = {"TP": "prediction = 1 and label = 1", "FP": "prediction = 1 and label = 0",
                             "FN": "prediction = 0 and label = 1", "TN": "prediction = 0 and label = 0"}
    category_conv_ids = {category: list() for category in categories}

    u_ids = [1, 2, 3, "dgpt", "gpt3"]
    labels = list()
    predictions = list()
    for key, conv in id_to_conv.items():
        for u_id in u_ids:
            label = conv.get_off_label(u_id)
            prediction = conv.get_off_prediction(u_id)
            if label is not None and prediction is not None:
                # keep this label and prediction
                if prediction == 1 and label == 1:
                    # TP
                    category_conv_ids["TP"].append((key, u_id))
                elif prediction == 1 and label == 0:
                    # FP
                    category_conv_ids["FP"].append((key, u_id))
                elif prediction == 0 and label == 1:
                    # FN
                    category_conv_ids["FN"].append((key, u_id))
                elif prediction == 0 and label == 0:
                    # TN
                    category_conv_ids["TN"].append((key, u_id))
                else:
                    # Incorrect prediction or label
                    logging.error(f"Incorrect prediction({prediction}) or label({label})")
                    exit(1)

    # Log a sample form each category
    for category in categories:
        if len(category_conv_ids[category]) <= K:
            sample_size = len(category_conv_ids[category])
        else:
            sample_size = K
        logging.info(
            f"{category}:{category_explanations[category]}:A sample of {sample_size}/{len(category_conv_ids[category])} instances:")
        category_sample = random.sample(category_conv_ids[category], sample_size)
        # print the conversations in this category with u_ids
        for key, u_id in category_sample:
            id_to_conv[key].log_offensive_prediction(u_id)


#########################################################################
########## Functions for entire sequence Stance Classification
#########################################################################

def create_stance_instances_from_convs(conversations_data, adjacent_only=False):
    instances = list()
    for conv in conversations_data:
        if conv.data_source == "OC_S":
            subreddit, sample_type, thread_id = conv.subreddit, conv.sample_type, conv.thread_id
            # Extract conversational utterances and off_labels
            utterances = list()

            # If stance flag is given then collect stance labels and stance u_id pairs
            stance_labels = list()
            stance_u_id_pairs = list()
            for i, current_u_data in enumerate(conv.utterance_data):
                if i == 0:
                    # First comment data is the post
                    comment = current_u_data["comment"][2:].strip()
                    comment = f"subreddit = {conv.subreddit} {comment}"
                    utterances.append(comment)
                else:
                    utterances.append(current_u_data["comment"][2:])
                u_id = current_u_data["id"]
                # Ignore the first utterance
                if u_id < 2:
                    continue
                # Find stance labels and u_id pairs for previous utterances and save them in lists
                for i in range(1, u_id):
                    # Skip the non-adjacent pairs if adjacent_only is given
                    if adjacent_only and i + 1 != u_id:
                        continue
                    u_id_pair = (i, u_id)
                    stance_label = current_u_data[f"{i}stance"]
                    stance_labels.append(normalize_stance_label(stance_label))
                    stance_u_id_pairs.append(u_id_pair)
            dgpt_utterances = utterances + [conv.dgpt_resp_data["comment"]]
            gpt3_utterances = utterances + [conv.gpt3_resp_data["comment"]]

            # Create stance labels and u_id pairs for DGPT and GPT3 responses
            n_thread_utterances = len(conv.utterance_data)
            dgpt_stance_labels = list()
            dgpt_stance_u_id_pairs = list()
            gpt3_stance_labels = list()
            gpt3_stance_u_id_pairs = list()
            u_id = n_thread_utterances + 1
            for i in range(1, n_thread_utterances + 1):
                # Skip the non-adjacent pairs if adjacent_only is given
                if adjacent_only and i != len(conv.utterance_data):
                    continue
                u_id_pair = (i, u_id)
                dgpt_stance_label = conv.dgpt_resp_data[f"{i}stance"]
                gpt3_stance_label = conv.gpt3_resp_data[f"{i}stance"]
                dgpt_stance_labels.append(normalize_stance_label(dgpt_stance_label))
                dgpt_stance_u_id_pairs.append(u_id_pair)
                gpt3_stance_labels.append(normalize_stance_label(gpt3_stance_label))
                gpt3_stance_u_id_pairs.append(u_id_pair)
            # Add the utterance stance with dgpt and gpt3 stance
            dgpt_stance_labels = stance_labels + dgpt_stance_labels
            dgpt_stance_u_id_pairs = stance_u_id_pairs + dgpt_stance_u_id_pairs
            gpt3_stance_labels = stance_labels + gpt3_stance_labels
            gpt3_stance_u_id_pairs = stance_u_id_pairs + gpt3_stance_u_id_pairs

            # Add everything to the instance list
            instances.append(
                {"conv": conv, "resp_type": "dgpt", "utterances": dgpt_utterances, "stance_labels": dgpt_stance_labels,
                 "stance_u_id_pairs": dgpt_stance_u_id_pairs})
            instances.append(
                {"conv": conv, "resp_type": "gpt3", "utterances": gpt3_utterances, "stance_labels": gpt3_stance_labels,
                 "stance_u_id_pairs": gpt3_stance_u_id_pairs})
        else:
            logging.error(f"Unrecognized data from source = {conv.data_source}")
            exit()
    return instances


class OC_S_stance_Dataset(Dataset):
    """OC_S_stance_Dataset stores the OC_S_post_thread instances. It takes list of Conversation_Data.
        It transforms the input list of Conversation_Data into list of dictionary instances"""

    def __init__(self, conversations_data, adjacent_only=False):
        super(OC_S_stance_Dataset, self).__init__()
        self.instances = create_stance_instances_from_convs(conversations_data, adjacent_only)
        self.nsamples = len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return self.nsamples


def log_TP_FP_FN_TN_convs_from_stance_predictions(id_to_conv, given_label=1, adjacent_only=False, K=5):
    categories = ["TP", "FP", "FN"]
    category_explanations = {"TP": f"prediction = {given_label} and label = {given_label}",
                             "FP": f"prediction = {given_label} and label = ?",
                             "FN": f"prediction = ? and label = {given_label}"}
    category_conv_ids = {category: list() for category in categories}

    u_id_pairs = [(1, 2), (1, 3), (2, 3), (1, "dgpt"), (2, "dgpt"), (3, "dgpt"), (1, "gpt3"), (2, "gpt3"), (3, "gpt3")]
    labels = list()
    predictions = list()
    for key, conv in id_to_conv.items():
        # Only log TP FP FN TN for given given_label
        for to_id, from_id in u_id_pairs:
            if adjacent_only:
                # Skip the current pair if not adjacent
                if type(from_id) == int and to_id != (from_id - 1):
                    continue
                if from_id in ["dgpt", "gpt3"] and to_id != len(conv.utterance_data):
                    continue
            label = normalize_stance_label(conv.get_stance_label(from_id, to_id))
            prediction = conv.get_stance_prediction(from_id, to_id)
            if label is not None and prediction is not None:
                if prediction != given_label and label != given_label:
                    # Possible TN
                    # TODO: Think about this later
                    continue
                else:
                    # keep this label and prediction
                    if prediction == label:
                        # TP
                        category_conv_ids["TP"].append((key, (to_id, from_id)))
                    elif prediction == given_label:
                        # FP
                        category_conv_ids["FP"].append((key, (to_id, from_id)))
                    elif label == given_label:
                        # FN
                        category_conv_ids["FN"].append((key, (to_id, from_id)))
                    else:
                        # Incorrect prediction or label
                        logging.error(f"Incorrect prediction({prediction}) or label({label})")
                        exit(1)

    # Log a sample form each category
    # Also save them in a csv for careful analysis
    analysis_csv_rows = list()
    for category in categories:
        if len(category_conv_ids[category]) <= K:
            sample_size = len(category_conv_ids[category])
        else:
            sample_size = K
        logging.info(
            f"{category}:{category_explanations[category]}:A sample of {sample_size}/{len(category_conv_ids[category])} instances:")
        analysis_csv_rows.append([category,
                                  f"{category_explanations[category]}:A sample of {sample_size}/{len(category_conv_ids[category])} instances:"])
        category_sample = random.sample(category_conv_ids[category], sample_size)
        # print the conversations in this category with u_ids
        for key, (to_id, from_id) in category_sample:
            current_example_analysis_rows = id_to_conv[key].log_stance_prediction(from_id, to_id)
            analysis_csv_rows.extend(current_example_analysis_rows)
        analysis_csv_rows.append([])
        logging.info("")
    return analysis_csv_rows


def log_top_conv_stance_predictions(id_to_conv, given_label=1, adjacent_only=False, K=5):
    u_id_pairs = [(1, 2), (1, 3), (2, 3), (1, "dgpt"), (2, "dgpt"), (3, "dgpt"), (1, "gpt3"), (2, "gpt3"), (3, "gpt3")]
    # TEMP changing this to only reddit comment replies
    u_id_pairs = [(1, 2), (1, 3), (2, 3)]
    labels = list()
    predictions = list()

    score_tracker = list()
    for key, conv in id_to_conv.items():
        # Keep track of the key u_id pairs and scores from prediciton
        for to_id, from_id in u_id_pairs:
            if adjacent_only:
                # Skip the current pair if not adjacent
                if type(from_id) == int and to_id != (from_id - 1):
                    continue
                if from_id in ["dgpt", "gpt3"] and to_id != len(conv.utterance_data):
                    continue
            label = normalize_stance_label(conv.get_stance_label(from_id, to_id))
            prediction = conv.get_stance_prediction(from_id, to_id)
            score = conv.get_stance_prediction_score(from_id, to_id)
            if label is not None and prediction is not None:
                label_score = score[given_label]
                score_tracker.append((key, (to_id, from_id), label_score))
    # Sort the keys based on the label_scores
    score_tracker_sorted = sorted(score_tracker, key=lambda tup: tup[2], reverse=True)
    # Log top K convs with highest scores
    # Also save them in a csv for careful analysis
    analysis_csv_rows = list()
    k = 0
    for key, (to_id, from_id), label_score in score_tracker_sorted:
        k += 1
        # Print the current conv with prediction
        current_example_analysis_rows = id_to_conv[key].log_stance_prediction(from_id, to_id)
        analysis_csv_rows.extend(current_example_analysis_rows)
        if k == K:
            break
        analysis_csv_rows.append([])
        logging.info("")
    return analysis_csv_rows


#########################################################################
########## Functions for Pairwise Stance Classification
#########################################################################
# We will have both cases
# 1 - All pairwise stance pairs
# 2 - Only adjacent pairwise stance pairs

def create_pairwise_stance_instances_from_convs(conversations_data, adjacent_only=False):
    instances = list()
    for conv in conversations_data:
        if conv.data_source == "OC_S":
            subreddit, sample_type, thread_id = conv.subreddit, conv.sample_type, conv.thread_id
            # Collect stance labels and stance u_id pairs from utterances
            for i, current_u_data in enumerate(conv.utterance_data):
                u_id = current_u_data["id"]
                # Ignore the first utterance
                if u_id < 2:
                    continue
                # Add stance label and their u pairs in the instances
                for i in range(1, u_id):
                    from_id = u_id
                    from_u = conv.get_processed_utterance_from_id(from_id)
                    to_id = i
                    to_u = conv.get_processed_utterance_from_id(to_id)
                    stance_label = current_u_data[f"{to_id}stance"]
                    stance_label = normalize_stance_label(stance_label)
                    # Skip the non-adjacent pairs if adjacent_only is given
                    if adjacent_only and i + 1 != u_id:
                        continue
                    # Add the instance to the final list
                    instances.append({"conv": conv, "to_id": to_id, "from_id": from_id, "to_u": to_u, "from_u": from_u,
                                      "stance_label": stance_label})
            # Add the stance pairs from DGPT and GPT3 responses
            for resp_data in [conv.dgpt_resp_data, conv.gpt3_resp_data]:
                # Add stance label and their u pairs in the instances
                for i in range(1, len(conv.utterance_data) + 1):
                    from_id = resp_data["id"]
                    from_u = conv.get_processed_utterance_from_id(from_id)
                    to_id = i
                    to_u = conv.get_processed_utterance_from_id(to_id)
                    stance_label = resp_data[f"{to_id}stance"]
                    stance_label = normalize_stance_label(stance_label)
                    # Skip the non-adjacent pairs if adjacent_only is given
                    if adjacent_only and i != len(conv.utterance_data):
                        continue
                    # Add the instance to the final list
                    instances.append({"conv": conv, "to_id": to_id, "from_id": from_id, "to_u": to_u, "from_u": from_u,
                                      "stance_label": stance_label})
        else:
            logging.error(f"Unrecognized data from source = {conv.data_source}")
            exit()
    return instances


class OC_S_pairwise_stance_Dataset(Dataset):
    """OC_S_pairwise_stance_Dataset stores stance pairs as instances. It takes list of Conversation_Data.
        It transforms the input list of Conversation_Data into list of dictionary instances"""

    def __init__(self, conversations_data, adjacent_only=False):
        super(OC_S_pairwise_stance_Dataset, self).__init__()
        self.instances = create_pairwise_stance_instances_from_convs(conversations_data, adjacent_only)
        self.nsamples = len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return self.nsamples
