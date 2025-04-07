def get_precision_recall_f1(predicts, test_data, tag_to_ix):
    good_tags = {}
    pred_tags = {}
    ground_tags = {}
    for i, predict in enumerate(predicts):
        gold = test_data[i][1]
        for i, tag in enumerate(predict):
            if tag == tag_to_ix[gold[i]]:
                if tag not in good_tags:
                    good_tags[tag] = 1
                else:
                    good_tags[tag] += 1
            if tag in pred_tags:
                pred_tags[tag] += 1
            else:
                pred_tags[tag] = 1
            if tag not in ground_tags:
                ground_tags[tag] = 1
            else:
                ground_tags[tag] += 1
        
    precision_scores = {}
    for tag in tag_to_ix:
        precision_scores[tag] = good_tags.get(tag, 0) / pred_tags.get(tag, 1)

    recall_scores = {}
    for tag in tag_to_ix:
        recall_scores[tag] = good_tags.get(tag, 0) / ground_tags.get(tag, 1)

    f1_scores = {}
    for tag in tag_to_ix:
        f1_scores[tag] = 2 * precision_scores[tag] * recall_scores[tag] / (precision_scores[tag] + recall_scores[tag] + 1e-10)

    return precision_scores, recall_scores, f1_scores
