from pyspark.sql.functions import col, udf, collect_list
from classifier import Classifier
from pet_udfs import has_animal
from pet_udfs import flattenUdf

def find_pet_owners(ddf, pet_list, keywords=['my','our','have','ive'], n=3):
    """ Claim pet ownership if within the comment is found one of the key_word
        in the n words preceding the pet (e.g. 'dog')
    """
    pet_owners_ddf = ddf. \
        select('creator_name',
               'userid',
               'words')
    for pet in pet_list:
        col_name = '{}_owner'.format(pet)
        pet_owners_ddf = pet_owners_ddf. \
            withColumn(col_name, has_animal(pet, keywords, n)(col('words')))
    condition = '_owner OR '.join(pet_list) + '_owner'
    return pet_owners_ddf.where(condition)


def build_pet_classifier(cat_dog_ddf, pet):
    print '\n{} classifier metrics'.format(pet)
    clf = Classifier(target_col=pet+'_owner',
                     prediction_col='predict_{}_owner'.format(pet))
    transformed_ddf = clf.fit_transform(cat_dog_ddf, token_col='words')
    clf.training(transformed_ddf)
    return clf


def predict_owners(dataset_ddf, trained_model):
    transformed_ddf = trained_model.transform(dataset_ddf,
                                              token_col='words')
    return trained_model.prediction(transformed_ddf)


def LDA_dataset_preparation(pet_owners_ddf):
    """ Collapse all the messages from owners of same pet in a single document"""
    corpus = pet_owners_ddf.groupby(['predict_dog_owner','predict_cat_owner']). \
        agg(collect_list('words').alias('words')). \
        withColumn('words', flattenUdf('words')). \
        where('(predict_dog_owner * predict_cat_owner) = 0'). \
        selectExpr('words',
                   '(CASE WHEN predict_dog_owner=1 THEN 1 '
                   'WHEN predict_cat_owner=1 THEN 2 '
                   'ELSE 0 END) AS pet_index')
    return corpus
