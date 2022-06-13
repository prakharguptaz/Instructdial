import string
from string import Template, ascii_uppercase

import settings

def get_options_string(classes):
    assert type(classes) == list

    return f'{settings.OPTION_TOKEN} ' + (settings.OPTION_SEPARATOR).join(classes)


def get_alphabetwithoptions_string(candidates):
    assert type(candidates) == list
    
    candidate_with_option = []
    for option, candidate in zip(ascii_uppercase, candidates):
        candidate_with_option.append(f'{option}: {candidate}')
    return f'{settings.OPTION_TOKEN} ' + (settings.OPTION_SEPARATOR).join(candidate_with_option)

def get_integerwithoptions_string(candidates):
    assert type(candidates) == list

    candidate_with_option = []
    for option_index, candidate in enumerate(candidates):
        candidate_with_option.append(f'{option_index}: {candidate}')
    return f'{settings.OPTION_TOKEN} ' + (settings.OPTION_SEPARATOR).join(candidate_with_option)