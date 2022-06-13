# Persuasion for Good

from sklearn.metrics import f1_score


# Persuasion for Good
def macro_f1(actual, predicted):
    return f1_score(actual, predicted, average='macro')

# Casino uses
# - F1
# - Joint-A: "measures the percentage of utterances for which the model predicts all the strategies correctly"


def test():
    print('empty')


if __name__ == "__main__":
    test()
