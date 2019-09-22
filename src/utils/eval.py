import os


def conlleval(label_path, metric_path):
    """
    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "../utils/conlleval_rev.pl"
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics
    