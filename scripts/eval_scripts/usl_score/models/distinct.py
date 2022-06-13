from collections import deque
from transformers import BertTokenizer

def distinct(responses, n_list=[1,2]):
    n_grams = { n: set() for n in n_list }
    n_freqs = { n: 0 for n in n_list }
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for response in responses:
        tokens = response.split()
        # tokens = tokenizer.tokenize(response)
        queues = { n: deque(maxlen=n) for n in n_list }

        for token in tokens:
            # total_tokens += 1
            for n in n_list:
                queue = queues[n] # queue corresponds to gram
                queue.append(token)

                if len(queue) == n:
                    gram = "__".join(list(queue))
                    n_grams[n].add(gram)
                    n_freqs[n] = n_freqs[n] + 1

    scores = {}
    for n, v in n_grams.items():
        distinct_score = len(v) / n_freqs[n]
        scores[f'distinct_{n}'] = distinct_score
    return scores

if __name__  == "__main__":
    responses = ['hi there how are you?', 'omg. do you think so too?']
    scores = distinct(responses, n_list=[1,2,3,4])
    print (scores)
