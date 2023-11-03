class UtilString:
    def __init__(self):
        pass

    @staticmethod
    def character_tokenize(s: str):
        tokens = []
        for c in s:
            tokens.append(c)
        return tokens

    @staticmethod
    def str_to_bool(s):
        s = s.lower()
        if s in ['true', 't', 'yes', 'y', '1']:
            return True
        elif s in ['false', 'f', 'no', 'n', '0']:
            return False
        else:
            return None