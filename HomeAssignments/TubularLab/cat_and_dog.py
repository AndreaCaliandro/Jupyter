from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import udf, col, array_contains, collect_list, desc
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, BooleanType, IntegerType, StringType

from pets import find_pet_owners, build_pet_classifier, predict_owners, LDA_dataset_preparation
from topic_extraction import TopicExtraction
from pet_udfs import significance_udf, flattenUdf

spark=SparkSession.builder.appName('cat_and_dog').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# path = '/Users/andreacaliandro/projects/Jupyter/HomeAssignments/TubularLab/data/animals_comments.csv'
path = './data/animals_comments.csv'


def read_and_tokenize(path):
    """ Reads the cvs file, and fills a ddf,
        clean the not-ascii characters,
        tokenize the comment"""
    to_ascii = udf(lambda s: s.encode("ascii", "ignore") if s else None, StringType())
    slen = udf(lambda s: len(s) if s else 0, IntegerType())

    ddf = spark.read.csv(path, inferSchema=True, header=True, encoding='ascii'). \
        select(to_ascii('creator_name').alias('creator_name'),
               'userid',
               to_ascii('comment').alias('comment')). \
        where(slen('comment')>0)

    tokenizer = Tokenizer(inputCol="comment", outputCol="words")
    tokenized = tokenizer.transform(ddf)
    return tokenized


if __name__ == '__main__':
    dataset_ddf = read_and_tokenize(path)

    # Step 1. Identify Cat And Dog Owners
    cat_dog_ddf = find_pet_owners(dataset_ddf, ['dog', 'cat', 'bird', 'turtle', 'horse']). \
        groupby('userid').agg(collect_list('words').alias('words'),
                              F.max('dog_owner').alias('dog_owner'),
                              F.max('cat_owner').alias('cat_owner')
                              ). \
        withColumn('words', flattenUdf('words')). \
        where('NOT (dog_owner AND cat_owner)'). \
        persist(StorageLevel.MEMORY_AND_DISK)
    print cat_dog_ddf.count()
    cat_dog_ddf.show(10)

    # Step 2: Build And Evaluate Classifiers
    # Dog owners and Cat owners classifiers
    dog_clf = build_pet_classifier(cat_dog_ddf, 'dog')
    cat_clf = build_pet_classifier(cat_dog_ddf, 'cat')

    # Step 3: Classify All The Users
    # Looking for dog and cat owners in whole dataset
    predictable_ddf = dataset_ddf. \
        groupBy('userid').agg(collect_list('words').alias('words')). \
        withColumn('words', flattenUdf('words')). \
        persist(StorageLevel.MEMORY_AND_DISK)
    dog_ddf = predict_owners(predictable_ddf, dog_clf)
    cat_ddf = predict_owners(predictable_ddf, cat_clf)
    pet_owners_ddf = dog_ddf.join(cat_ddf, 'userid'). \
        select(dog_ddf['userid'],
               dog_ddf['words'],
               dog_ddf['predict_dog_owner'],
               cat_ddf['predict_cat_owner'])
    pet_owners_ddf.show()

    # Step 4: Extract Insights About Cat And Dog Owners
    # Topic extraction with LDA
    docs_ddf = LDA_dataset_preparation(pet_owners_ddf.limit(1000)). \
        persist(StorageLevel.DISK_ONLY)
    # docs_ddf.show()
    topics = TopicExtraction(docs_ddf, topic_num=10)
    topics.process()

    # Step 5: Identify Creators With Cat And Dog Owners In The Audience
    creators_ddf = dataset_ddf.join(pet_owners_ddf, 'userid'). \
        select('creator_name',
               pet_owners_ddf['userid'],
               'predict_dog_owner',
               'predict_cat_owner'). \
        groupby('creator_name').sum('predict_dog_owner', 'predict_cat_owner'). \
        withColumn('dog_count', col('sum(predict_dog_owner)')). \
        withColumn('cat_count', col('sum(predict_cat_owner)')). \
        selectExpr('creator_name',
                   'dog_count',
                   'cat_count',
                   'dog_count*dog_count AS dog_count2',
                   'cat_count*cat_count AS cat_count2')
    creators_ddf.show()

    dog_m, dog2, cat_m, cat2 = creators_ddf.groupby().avg('dog_count', 'dog_count2',
                                                          'cat_count', 'cat_count2').collect()[0]
    ranking_ddf = creators_ddf. \
        withColumn('dog_significance', significance_udf(dog_m, dog2)(col('dog_count'))). \
        withColumn('cat_significance', significance_udf(cat_m, cat2)(col('cat_count')))

    print('Top 10 creators with higher number of dog owners')
    ranking_ddf.select('creator_name',
                       'dog_count',
                       'dog_significance'). \
        orderBy(desc('dog_significance')).show(10)

    print('Top 10 creators with higher number of cat owners')
    ranking_ddf.select('creator_name',
                       'cat_count',
                       'cat_significance'). \
        orderBy(desc('cat_significance')).show(10)

