package com.upgrad.ajbose;

import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.tree.DecisionTreeParams;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;


public class classification {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession sparkSession = SparkSession.builder()  //SparkSession  
                .appName("SparkML") 
                .master("local[*]") 
                .getOrCreate(); //
      
        Dataset<Row> userKnowDf = sparkSession.read().option("header", true).option("inferSchema",true).csv("data/UserKnowModelingDataset_Train.csv");
        //Reading Data from a CSV file //Inferring Schema and Setting Header as True
        
        userKnowDf.show(); //Displaying Samples
        userKnowDf.printSchema(); //Printing Schema
        userKnowDf.describe().show(); // Statistically Summarizing about the data

        
        //**************************************String Indexer***************************************************//
        
        
		StringIndexer indexer = new StringIndexer().setInputCol("SKL").setOutputCol("IND_SKL");

		StringIndexerModel indModel = indexer.fit(userKnowDf);
		Dataset<Row> indexedUserKnow = indModel.transform(userKnowDf);
		indexedUserKnow.groupBy(col("SKL"), col("IND_SKL")).count().show();
		
		
		//**********************************Assembling the vector and label************************//
		
		
		Dataset<Row> df= indexedUserKnow.select(col("IND_SKL").as("label"),col("SST"),col("SRT"),col("SAT"),col("SAP"),col("SEP"));
        //df.show();

        //Assembling the features in the dataFrame as Dense Vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"SST","SRT","SAT","SAP","SEP"})
                .setOutputCol("features");
        
        Dataset<Row> LRdf = assembler.transform(df).select("label","features");    
        LRdf.show();
 
        
        //*****************************Model Building *****************************************//
        
		Dataset<Row>[] splits = LRdf.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];		//Training Data
        Dataset<Row> testData = splits[1];			//Testing Data
        
    	DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setSeed(0);

    	DecisionTreeClassificationModel Model = dt.fit(trainingData);		

    	System.out.println("Learned Decision tree" + Model.toDebugString());
		// Predict on test data
    	
		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString().setInputCol("label").setOutputCol("labelStr")
				.setLabels(indModel.labels());

		IndexToString predConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictionStr")
				.setLabels(indModel.labels());

		Dataset<Row> rawPredictions = Model.transform(testData);
		//rawPredictions.show();
		
		Dataset<Row> predictions = predConverter.transform(labelConverter.transform(rawPredictions));
		//predictions.show();
		predictions.select("predictionStr", "labelStr", "features").show(5);

		/*************************Model Evaluation*********************/
		// View confusion matrix
		System.out.println("Confusion Matrix :");
		predictions.groupBy(col("labelStr"), col("predictionStr")).count().show();

		// Accuracy computation
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy"); 
		double accuracy = evaluator.evaluate(predictions);
		System.out.println("Accuracy = " + Math.round(accuracy * 100) + " %");

	}

}
