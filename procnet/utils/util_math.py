class UtilMath:
    def __init__(self):
        pass

    @staticmethod
    def calculate_f1(p, r):
        if p + r == 0:
            return 0
        f1 = (2 * p * r) / (p + r)
        return f1

    @staticmethod
    def calculate_precision_recall_f1(true_positive, ans_positive, pred_positive):
        precision = true_positive / pred_positive if pred_positive != 0 else 0
        recall = true_positive / ans_positive if ans_positive != 0 else 0
        f1 = UtilMath.calculate_f1(precision, recall)
        return precision, recall, f1
