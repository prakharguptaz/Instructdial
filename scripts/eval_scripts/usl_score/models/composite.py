
def composite(u_scores, s_scores, l_scores, method='HS', coef=[0.33, 0.33, 0.34]):
    for u,s,l in zip(u_scores, s_scores, l_scores):
        score = composite_one_instance(u, s, l, method=method, coef=coef)
        scores.append(score)
    return scores

def composite_one_instance(u,s,l, method='HS', coef=[0.33, 0.33, 0.34]):
    if method == 'A':
        score = coef[0]*u + coef[1]*s + coef[2]*l
    elif method == 'H':
        score = coef[0]*u + coef[1]*u*s + coef[2]*u*s*l
    elif method == 'HS':
        score = coef[0]*u + coef[1]*s + coef[2]*s*l
    return score
