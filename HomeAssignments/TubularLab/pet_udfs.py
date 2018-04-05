from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, BooleanType, DoubleType

from math import sqrt

def has_close_word(tokens, subject, key_words, n):
    if (subject in tokens) is False:
        return False
    else:
        for word in key_words:
            if (word in tokens):
                dist = tokens.index(subject) - tokens.index(word)
                return True if (dist < n and dist > 0) else False
            else:
                return False

def has_animal(animal, key_words, n):
    return udf(lambda col: has_close_word(col, animal, key_words, n), BooleanType())


def fudf(val):
    return reduce (lambda x, y:x+y, val)

flattenUdf = udf(fudf, ArrayType(StringType()))


def significance_udf(mean, var):
    return udf(lambda col: (col - mean) / (sqrt(var - mean ** 2)), DoubleType())
