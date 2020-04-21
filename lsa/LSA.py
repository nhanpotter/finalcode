from __future__ import print_function

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer


def append(arr):
    tmp = []
    for x in arr:
        temp = x.split('.')
        if len(temp) < 3:
            for n in range(len(temp), 3):
                temp.append('')
        tmp.append(temp)
    return tmp


def get_lsa_score(model_answer_sentences, student_answer_sentences):
    # Apply SVD
    vectorizer = CountVectorizer()
    try:
        dtm = vectorizer.fit_transform(student_answer_sentences)
    except:
        return np.nan
    vectorizer2 = CountVectorizer()
    try:
        dtm2 = vectorizer2.fit_transform(model_answer_sentences)
    except:
        return np.nan

    if dtm.shape[1] < 3 or dtm2.shape[1] < 3:
        return np.nan

    lsa = TruncatedSVD(2, algorithm='arpack')
    dtm_lsa = lsa.fit_transform(dtm.astype(float))
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    lsa2 = TruncatedSVD(2, algorithm='arpack')
    dtm_lsa_2 = lsa2.fit_transform(dtm2.astype(float))
    dtm_lsa_2 = Normalizer(copy=False).fit_transform(dtm_lsa_2)

    # Compute document similarity using LSA components
    similarity_matrix = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa_2).T)

    # Calculate final score
    passed = 0
    for vec in similarity_matrix:
        for val in vec:
            if val > 0.8:
                passed += 1
                break
    similarity = passed / len(similarity_matrix)
    return similarity
