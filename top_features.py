def top_features_by_class(tfidf_mat, features):

    for doc in tfidf_mat:
        doc

    return features


def top_features_by_doc(tfidf_mat, features):

    for doc in tfidf_mat:
        doc

    return features

def to_lower(gene_set):
    s = ' '.join(gene_set)
    s=s.lower()

    l = s.split(' ')

    return set(l)




def set_cl_probs(r, gene_class_probs):
    g = r['Gene']
    for i in range(1,9):
        if i in gene_class_probs[g]:
            r['cl'+str(i)] = gene_class_probs[g][i]

    return r