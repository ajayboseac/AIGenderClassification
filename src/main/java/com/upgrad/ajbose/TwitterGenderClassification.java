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

import java.text.SimpleDateFormat;
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

    public static void main(String[] args) {

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


        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{ "created", "fav_number", "link_color",
                        "sidebar_color", "tweet_count", "tweet_created", "ind_user_timezone","textFeatures"})
                .setOutputCol("features");


        Dataset<Row> finalData = assembler.transform(dataAfterPreprocessing).select("label","features");


        Dataset<Row>[] splits = finalData.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];		//Training Data
        Dataset<Row> testData = splits[1];			//Testing Data
        // Fit the pipeline to training documents.

        DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setSeed(0);
        dt.setMaxBins(160);

        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel").setLabels(genderIndexerModel.labels());

        Pipeline modelExecutionPipeLine = new Pipeline().setStages(new PipelineStage[]{dt, labelConverter});


        PipelineModel modelExecutionPipeLineModel = modelExecutionPipeLine.fit(trainingData);
        // Make predictions on training documents.
        Dataset<Row> predictions = modelExecutionPipeLineModel.transform(testData);

        predictions.groupBy(col("label"),col("prediction"),col("predictedLabel")).count().show();

        // Accuracy computation
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
                .setPredictionCol("prediction");


        double fscore = evaluator.evaluate(predictions);
        System.out.println("fscore = " + fscore );



        Dataset<Row> predictions2 = modelExecutionPipeLineModel.transform(trainingData);

        predictions2.groupBy(col("label"),col("prediction"),col("predictedLabel")).count().show();

        // Accuracy computation
        MulticlassClassificationEvaluator evaluator2 = new MulticlassClassificationEvaluator().setLabelCol("label")
                .setPredictionCol("prediction");


        double fscore2 = evaluator.evaluate(predictions2);
        System.out.println("fscore = " + fscore2 );

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


        // Create and Run Random Forest Pipeline
        Pipeline dataPreProcessingPipeLine = new Pipeline()
                .setStages(new PipelineStage[] {timeZoneIndexerModel,genderIndexerModel, tokenizer, remover, hashingTF, idf});

        PipelineModel pipelineModel = dataPreProcessingPipeLine.fit(cleanedData);
        return pipelineModel.transform(cleanedData);
    }

    private static void declareMethods(SparkSession sparkSession) {
        sparkSession.sqlContext().udf().register("dateConverterUDF", dateConverterUDF(), DataTypes.LongType);
        sparkSession.sqlContext().udf().register("colorCodeConverter", colorCodeConverter(), DataTypes.IntegerType);
        sparkSession.sqlContext().udf().register("timeZoneNullHandler", timeZoneNullHandler(), DataTypes.StringType);
    }

    private static Dataset<Row> cleanAndLoadData(SparkSession sparkSession) {
        //read the file as data
        String pathTrain = "src/main/resources/twitter_data.csv";
        Dataset<Row> data = sparkSession.read().format("csv").option("header", "true").load(pathTrain);

        Dataset<Row> df2 = data.selectExpr("gender", "created", "cast(fav_number as double ) fav_number", "link_color",
                "sidebar_color", "text", "cast(tweet_count as double ) tweet_count", "tweet_created", "user_timezone");

        //time to epochs
        Dataset<Row> df3 = df2.withColumn("created", functions.callUDF("dateConverterUDF", df2.col("created")));
        Dataset<Row> df4 = df3.withColumn("tweet_created", functions.callUDF("dateConverterUDF", df3.col("tweet_created")));

        //colors to integers
        Dataset<Row> df5 = df4.withColumn("sidebar_color", functions.callUDF("colorCodeConverter", df4.col("sidebar_color")));
        Dataset<Row> df6 = df5.withColumn("link_color", functions.callUDF("colorCodeConverter", df5.col("link_color")));
        Dataset<Row> df7 = df6.withColumn("user_timezone", functions.callUDF("timeZoneNullHandler", df6.col("user_timezone")));
        Dataset<Row> df8 = df7.where(col("gender").isin("male","female","brand"));

        //drop rows where gender is not present.
        return df8.na().drop();
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
