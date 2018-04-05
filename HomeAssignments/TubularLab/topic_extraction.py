from pyspark.ml.feature import StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark import StorageLevel

class TopicExtraction():

    def __init__(self, docs_ddf, topic_num, **model_params):
        self.docs_ddf = docs_ddf
        self.topic_num = topic_num

    def transform(self, token_col='words'):
        """ StopRemover, CountWords"""
        remover = StopWordsRemover(inputCol=token_col, outputCol="clean_words")
        clean_docs_ddf = remover.transform(self.docs_ddf)

        cv = CountVectorizer(inputCol="clean_words", outputCol="tf_vector")
        self.cv_model = cv.fit(clean_docs_ddf)
        self.word_counts_ddf = self.cv_model.transform(clean_docs_ddf). \
            persist(StorageLevel.DISK_ONLY)

    def extract_topics(self):
        lda = LDA(k=self.topic_num, featuresCol="tf_vector")
        self.model = lda.fit(self.word_counts_ddf)
        self.topics_ddf = self.model.transform(self.word_counts_ddf). \
            select('pet_index', 'topicDistribution')

    def show_topics(self):
        vocabulary = self.cv_model.vocabulary
        topics = self.model.describeTopics(5). \
            withColumn('terms', index2term(vocabulary)(col('termIndices'))). \
            select('topic', 'terms')
        print("The topics described by their top-weighted terms:")
        topics.show(truncate=False)
        print("Top topics each set of pet owners discuss about")
        self.topics_ddf. \
            withColumn('top_topics', top_topics(top_n=5)(col('topicDistribution'))). \
            selectExpr('(CASE '
                       'WHEN pet_index=1 THEN "dog_owners" '
                       'WHEN pet_index=2 THEN "cat_owners" '
                       'ELSE  "others" END) AS pet_owners',
                       'top_topics'). \
            show(truncate=False)

    def process(self):
        self.transform()
        self.extract_topics()
        self.show_topics()



# udfs
def term_list(index_list, vocabulary):
    return [vocabulary[i] for i in index_list]

def index2term(vocabulary):
    return udf(lambda col: term_list(col, vocabulary), ArrayType(StringType()))


def top(topic, top_n):
    return sorted(range(len(topic)), key=lambda i: topic[i], reverse=True)[:top_n]

def top_topics(top_n):
    return udf(lambda col: top(col, top_n), ArrayType(IntegerType()))