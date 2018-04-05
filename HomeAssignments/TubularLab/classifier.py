from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

class Classifier():

    def __init__(self, prediction_col, target_col=None):
        self.prediction_col = prediction_col
        self.target_col = target_col
        self.cv_model = None
        self.idf_model = None

    def fit_transform(self, tokenized_ddf, token_col='words', fit=True):
        if fit:
            tokenized_ddf = tokenized_ddf. \
                withColumn(self.target_col, col(self.target_col).cast(IntegerType()))
            cv = CountVectorizer(inputCol=token_col, outputCol="tf_vector")
            self.cv_model = cv.fit(tokenized_ddf)
        if self.cv_model is None:
            raise Exception('Error: cv_model is None.'
                            'The class needs to be previously fed with a training dataset')
        else:
            word_counts_ddf = self.cv_model.transform(tokenized_ddf)

        if fit:
            idf = IDF(inputCol="tf_vector", outputCol="tfidf_vector")
            self.idf_model = idf.fit(word_counts_ddf)
        return self.idf_model.transform(word_counts_ddf)

    def transform(self, tokenized_ddf, token_col='words'):
        return self.fit_transform(tokenized_ddf, token_col, fit=False)

    def add_weights(self):
        pass

    def training(self, transformed_ddf):
        train_ddf, test_ddf = transformed_ddf.randomSplit([0.7, 0.3])
        nb = NaiveBayes(smoothing=1.0, modelType="multinomial",
                        featuresCol="tfidf_vector",
                        labelCol=self.target_col,
                        predictionCol=self.prediction_col)
        self.model = nb.fit(train_ddf)
        self.evaluation(train_ddf, 'Train')
        self.evaluation(test_ddf, 'Test')

    def prediction(self, ddf):
        return self.model.transform(ddf)

    def evaluation(self, ddf, ddf_label='Test'):
        evaluator = MulticlassClassificationEvaluator(labelCol=self.target_col,
                                                      predictionCol=self.prediction_col)
        predictions = self.prediction(ddf)
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        print '{ddf_label} set accuracy = {metrics}'.format(ddf_label=ddf_label,
                                                            metrics=accuracy)
        fscore = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        print '{ddf_label} set F1 score = {metrics}'.format(ddf_label=ddf_label,
                                                            metrics=fscore)
