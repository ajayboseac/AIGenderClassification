package com.upgrad.ajbose;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import static org.apache.spark.sql.functions.col;

/**
 * Course Project as part of UpGrad's Big Data Engineering Program Course#6.
 * Here we are trying to predict the gender of a twitter account holder using the account's
 * behavior data into three Categories.
 * <p>
 * 1. Male
 * 2. Female
 * 3. Brand
 * <p>
 * We need to pre-process the data and fix data issues if there are any. Before we get to
 * building the models. Issues like missing values needs to be fixed.
 * <p>
 * We will build two kind of models.
 * <p>
 * 1. Decision Tree Classifier Model
 * 2. Random Forest Classifier Model
 * <p>
 * Once the models are built we will use various metrics to calculate how the models are
 * performing. We will use the beow metrics
 * <p>
 * 1. Evaluation Scores: Accuracy, Precision, Recall, F1 Score.
 * 2. Confusion Matrix.
 */
public class TwitterGenderClassification {

    static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("mm/dd/yy HH:MM");

    public static void main(String[] args) throws IOException {

        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession sparkSession = SparkSession.builder()  //SparkSession
                .appName("TwitterGenderClassification")
                .master("local[*]")
                .getOrCreate(); //

        declareMethods(sparkSession);

        Dataset<Row> cleanedData = cleanAndLoadData(sparkSession);


        StringIndexerModel genderIndexerModel = new StringIndexer().setInputCol("gender").setOutputCol("label").fit(cleanedData);
        genderIndexerModel.setHandleInvalid("keep");

        Dataset<Row> dataAfterPreprocessing = preProcessData(cleanedData, genderIndexerModel);
        dataAfterPreprocessing.show(10);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{
                        "link_color",
                        "sidebar_color",
                        "tweet_count",
                        "textFeatures",
                        "description_textFeatures",
                        "ind_user_timezone",
                        "tweet_created_day_of_week",
                        "tweet_created_month_of_year",
                        "tweet_created_hour_of_day" ,
                        "tweet_created_epoch",
                        "tweet_created_day_of_year",
                        "account_created_month_of_year",
                        "account_created_hour_of_day",
                        "account_created_epoch",
                        "account_created_day_of_year",
                        "account_created_day_of_week"
                })
                .setOutputCol("features");

        Dataset<Row> finalData = assembler.transform(dataAfterPreprocessing).select("label","features");


        Dataset<Row>[] splits = finalData.randomSplit(new double[]{0.7, 0.3},46l);
        Dataset<Row> trainingData = splits[0];		//Training Data
        Dataset<Row> testData = splits[1];//Testing Data


        // Fit the pipeline to training documents.

        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel").setLabels(genderIndexerModel.labels());
        decisionTreeModel(trainingData, testData, labelConverter);

        /* Decision Tree Classification */

        randomForest(trainingData,testData, labelConverter);

    }

    private static void randomForest(Dataset<Row> trainingData,Dataset<Row> testData, IndexToString labelConverter) {
        RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features");
        rf.setMaxBins(160);

        rf.setMaxDepth(20);
        rf.setMinInstancesPerNode(70);

        Pipeline rfmodelExecutionPipeLine = new Pipeline().setStages(new PipelineStage[]{rf, labelConverter});


        PipelineModel rfmodelExecutionPipeLineModel = rfmodelExecutionPipeLine.fit(trainingData);
        // Make predictions on training documents.
        Dataset<Row> testPredictions = rfmodelExecutionPipeLineModel.transform(testData);
        Dataset<Row> trainingtPredictions = rfmodelExecutionPipeLineModel.transform(trainingData);

        testPredictions.groupBy(col("label"),col("prediction"),col("predictedLabel")).count().show();
        evaluteModel(testPredictions,"RandomForestTest");
        evaluteModel(trainingtPredictions,"RandomForestTraining");

    }


    private static void decisionTreeModel(Dataset<Row> trainingData, Dataset<Row> testData, IndexToString labelConverter) {
        /* Decision Tree Classification */

        DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setSeed(0);
        dt.setMaxBins(160);


        Pipeline modelExecutionPipeLine = new Pipeline().setStages(new PipelineStage[]{dt, labelConverter});


        PipelineModel modelExecutionPipeLineModel = modelExecutionPipeLine.fit(trainingData);
        // Make predictions on training documents.
        Dataset<Row> testPredictions = modelExecutionPipeLineModel.transform(testData);
        Dataset<Row> trainingtPredictions = modelExecutionPipeLineModel.transform(trainingData);

        testPredictions.groupBy(col("label"),col("prediction"),col("predictedLabel")).count().show();


        evaluteModel(testPredictions,"DecisionTreeTest");
        evaluteModel(trainingtPredictions,"DecisionTreeTraining");
    }

    private static void evaluteModel(Dataset<Row> rfPredictions,String modelName) {
        // Accuracy computation
        MulticlassClassificationEvaluator rfEvaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
                .setPredictionCol("prediction");

        // Accuracy computation
        MulticlassClassificationEvaluator f1evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
                .setPredictionCol("prediction");
        MulticlassClassificationEvaluator precisionEvaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
                .setPredictionCol("prediction").setMetricName("weightedPrecision");
        MulticlassClassificationEvaluator recallEvaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
                .setPredictionCol("prediction").setMetricName("weightedRecall");

        double RFfscore = f1evaluator.evaluate(rfPredictions);
        double RFprecision = precisionEvaluator.evaluate(rfPredictions);
        double RFrecall = recallEvaluator.evaluate(rfPredictions);
        System.out.println(modelName+" f1score = " + RFfscore );
        System.out.println(modelName+" precision = " + RFprecision );
        System.out.println(modelName + " recall = " + RFrecall );
    }



    private static Dataset<Row> preProcessData(Dataset<Row> cleanedData, StringIndexerModel genderIndexerModel) {
        StringIndexerModel timeZoneIndexerModel = new StringIndexer().setInputCol("user_timezone").setOutputCol("ind_user_timezone").fit(cleanedData);
        timeZoneIndexerModel.setHandleInvalid("keep");


        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("text")
                .setOutputCol("words");

        // Remove the stop words
        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("filtered");

        // Create the Term Frequency Matrix
        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(remover.getOutputCol())
                .setOutputCol("numFeatures");

        // Calculate the Inverse Document Frequency
        IDF idf = new IDF()
                .setInputCol(hashingTF.getOutputCol())
                .setOutputCol("textFeatures");


        Tokenizer descriptionTokenizer = new Tokenizer()
                .setInputCol("description")
                .setOutputCol("description_words");

        // Remove the stop words
        StopWordsRemover descriptionRemover = new StopWordsRemover()
                .setInputCol(descriptionTokenizer.getOutputCol())
                .setOutputCol("description_filtered");

        // Create the Term Frequency Matrix
        HashingTF description_hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(descriptionRemover.getOutputCol())
                .setOutputCol("description_numFeatures");

        // Calculate the Inverse Document Frequency
        IDF descirption_idf = new IDF()
                .setInputCol(description_hashingTF.getOutputCol())
                .setOutputCol("description_textFeatures");



        // Create and Run Random Forest Pipeline
        Pipeline dataPreProcessingPipeLine = new Pipeline()
                .setStages(new PipelineStage[] {timeZoneIndexerModel,genderIndexerModel, tokenizer, remover,
                        hashingTF, idf,descriptionTokenizer,descriptionRemover,description_hashingTF,descirption_idf});

        PipelineModel pipelineModel = dataPreProcessingPipeLine.fit(cleanedData);
        Dataset<Row> preProcessedData = pipelineModel.transform(cleanedData);
        return preProcessedData;
    }

    private static void declareMethods(SparkSession sparkSession) {
        sparkSession.sqlContext().udf().register("dateConverterUDF", dateConverterUDF(), DataTypes.LongType);
        sparkSession.sqlContext().udf().register("colorCodeConverter", colorCodeConverter(), DataTypes.IntegerType);
        sparkSession.sqlContext().udf().register("timeZoneNullHandler", timeZoneNullHandler(), DataTypes.StringType);
        sparkSession.sqlContext().udf().register("dateToDayOfWeek", dateToDayOfWeek(), DataTypes.IntegerType);
        sparkSession.sqlContext().udf().register("dateToHourOfDay", dateToHourOfDay(), DataTypes.IntegerType);
        sparkSession.sqlContext().udf().register("dateToMonthOfYear", dateToMonthOfYear(), DataTypes.IntegerType);
        sparkSession.sqlContext().udf().register("dateToDayOfYear", dateToDayOfYear(), DataTypes.IntegerType);
    }

    private static Dataset<Row> cleanAndLoadData(SparkSession sparkSession) throws IOException {
        //read the file as data
        String pathTrain = "src/main/resources/twitter_data.csv";
        String pathAfterRemovingNewLines = "src/main/resources/output.csv";

        FileProcessor.handleNewLines(pathTrain);

        Dataset<Row> data = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("sep",",")
                .option("quote","\"")
                .option("multiline",true)
                .load(pathAfterRemovingNewLines);


        Dataset<Row> df2 = data.selectExpr("gender", "cast(fav_number as double ) fav_number", "link_color",
                "sidebar_color", "text", "cast(tweet_count as double ) tweet_count","created","tweet_created","user_timezone","description");

        Dataset<Row> df2_1 = df2.withColumn("user_timezone", functions.callUDF("timeZoneNullHandler", df2.col("user_timezone")));

        df2_1.na().drop();

        //time to epochs
        Dataset<Row> df3_created = df2_1.withColumn("account_created_epoch", functions.callUDF("dateConverterUDF", df2_1.col("created")));
        Dataset<Row> df4_created = df3_created.withColumn("account_created_hour_of_day", functions.callUDF("dateToHourOfDay", df3_created.col("created")));
        Dataset<Row> df5_created = df4_created.withColumn("account_created_month_of_year", functions.callUDF("dateToMonthOfYear", df4_created.col("created")));
        Dataset<Row> df6_created = df5_created.withColumn("account_created_day_of_week", functions.callUDF("dateToDayOfWeek", df5_created.col("created")));
        Dataset<Row> df7_created = df6_created.withColumn("account_created_day_of_year", functions.callUDF("dateToDayOfYear", df6_created.col("created")));


        // Converting here for tweet date.
        Dataset<Row> df4_tweet = df7_created.withColumn("tweet_created_epoch", functions.callUDF("dateConverterUDF", df7_created.col("tweet_created")));
        Dataset<Row> df5_tweet = df4_tweet.withColumn("tweet_created_hour_of_day", functions.callUDF("dateToHourOfDay", df4_tweet.col("tweet_created")));
        Dataset<Row> df6_tweet = df5_tweet.withColumn("tweet_created_month_of_year", functions.callUDF("dateToMonthOfYear", df5_tweet.col("tweet_created")));
        Dataset<Row> df7_tweet = df6_tweet.withColumn("tweet_created_day_of_week", functions.callUDF("dateToDayOfWeek", df6_tweet.col("tweet_created")));
        Dataset<Row> df8_tweet = df7_tweet.withColumn("tweet_created_day_of_year", functions.callUDF("dateToDayOfYear", df6_tweet.col("tweet_created")));
//        Dataset<Row> df4 =df3;

        df8_tweet.show(10);

        //colors to integers
        Dataset<Row> df8 = df8_tweet.withColumn("sidebar_color", functions.callUDF("colorCodeConverter", df8_tweet.col("sidebar_color")));
        Dataset<Row> df9 = df8.withColumn("link_color", functions.callUDF("colorCodeConverter", df8.col("link_color")));

        Dataset<Row> df10 = df9.where(col("gender").isin("male","female","brand"));

        return df10.na().drop();
    }

    private static UDF1<String, Long> dateConverterUDF() {
        return (String timeStamp) -> {
            if (timeStamp == null || timeStamp.isEmpty()) {
                return 0l;
            }
            try {
                Date date = DATE_FORMAT.parse(timeStamp);
                return date.getTime();
            }catch (Throwable e){
                return 0l;
            }
        };
    }

    private static UDF1<String, Integer> dateToDayOfWeek() {
        return (String timeStamp) -> {
            if (timeStamp == null || timeStamp.isEmpty()) {
                return 0;
            }
            try {
                Date date = DATE_FORMAT.parse(timeStamp);
                Calendar instance = Calendar.getInstance();
                instance.setTime(date);
                return instance.get(Calendar.DAY_OF_WEEK);
            }catch (Throwable e){
                return 0;
            }
        };
    }

    private static UDF1<String, Integer> dateToHourOfDay() {
        return (String timeStamp) -> {
            if (timeStamp == null || timeStamp.isEmpty()) {
                return 0;
            }
            try {
                Date date = DATE_FORMAT.parse(timeStamp);
                Calendar instance = Calendar.getInstance();
                instance.setTime(date);
                return instance.get(Calendar.HOUR_OF_DAY);
            }catch (Throwable e){
                return 0;
            }
        };
    }

    private static UDF1<String, Integer> dateToMonthOfYear() {
        return (String timeStamp) -> {
            if (timeStamp == null || timeStamp.isEmpty()) {
                return 0;
            }
            try {
                Date date = DATE_FORMAT.parse(timeStamp);
                Calendar instance = Calendar.getInstance();
                instance.setTime(date);
                return instance.get(Calendar.MONTH);
            }catch (Throwable e){
                return 0;
            }
        };
    }


    private static UDF1<String, Integer> dateToDayOfYear() {
        return (String timeStamp) -> {
            if (timeStamp == null || timeStamp.isEmpty()) {
                return 0;
            }
            try {
                Date date = DATE_FORMAT.parse(timeStamp);
                Calendar instance = Calendar.getInstance();
                instance.setTime(date);
                return instance.get(Calendar.DAY_OF_YEAR);
            }catch (Throwable e){
                return 0;
            }
        };
    }


    private static UDF1<String, String> timeZoneNullHandler() {
        return (String timeZone) -> {
            if (timeZone == null || timeZone.isEmpty()) {
                return "Empty";
            }
            return timeZone;
        };
    }


    private static UDF1<String, Integer> colorCodeConverter() {
        return (String colorCode) -> {
            if (colorCode == null || colorCode.isEmpty()) {
                return 0;
            }
            try {
                int colorCodeInteger = Integer.parseInt(colorCode,16);
                return colorCodeInteger;
            }catch (Throwable e){
                return 0;
            }
        };
    }


}
