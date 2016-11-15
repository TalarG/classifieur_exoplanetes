package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}
import org.apache.spark.ml.evaluation.RegressionEvaluator


/**
  * Created by Talar Guzelbodur and Severine Cohard on 27/10/2016.
  */

// TP 4-5 : Machine Learning avec spark

object JobML {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext



    // Etape initiale: chargement du fichier csv avec les données sauvegardées et nettoyées lors du TP précédent 

    val df = spark.read.parquet("/Users/tg104460/Downloads/SPARK/cleanedDataFrame.parquet/part-r-00000-48a00cdb-c844-4484-80ef-4e74b59ad949.snappy.parquet")

    // QUESTION 1: mise sous une forme utilisable dans Spark.ML des données  
    // A) mise en forme des colonnes 
    // 1ere étape : suppression des colonnes "koi_disposition" et "rowid"
    val df2 = df.drop("koi_disposition", "rowid").columns



    // 2ème étape: Construction de la colonne Features, qui contient des vecteurs, réunissant toutes les colonnes.
    val assembler = new VectorAssembler()
      .setInputCols(df2)
      .setOutputCol("features")


    val output = assembler.transform(df)
    println(output.select("features", "koi_disposition").first())

    // B) Transformation de la colonne des labels qui est une colonne de Strings (“CONFIRMED” ou “FALSE-POSITIVE”), en une colonne de 0 et de 1,
    //pour pouvoir faire une classification binaire.
    val indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("label")

    val indexed = indexer.fit(output).transform(output)
    indexed.show()


    //  QUESTION 2: Machine Learning
    // A) Splitter des données en "Training data" (90%) et "Test data"(10%), Le premier pour entrainer le modèle et le deuxième pour le tester.

    val Array(trainingData, testData) = indexed.randomSplit(Array(0.9, 0.1))

    // B) Entraînement du classifieur et réglage des hyper-paramètres de l’algorithme
    //Le modèle  de LogisticRegression:

    val  lr = new LogisticRegression()
      .setElasticNetParam(1.0)  // L1-norm regularization : LASSO
      .setLabelCol("label")
      .setStandardization(true)  // Pour mettre à l'échelle chaque feature du modèle
      .setFitIntercept(true)  // Regression affine
      .setTol(1.0e-5)  // critère d'arrêt de l'algorithme basé sur sa convergence
      .setMaxIter(300)  // critère d'arrêt de sécurité pour éviter des boucles infinies



    //Creation d'une grille de valeurs à tester pour les hyper-paramètres
    val puiss = - 6.0 to 0.0 by 0.5 toArray
    val puissance = puiss.map(n => scala.math.pow(10, n))
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, puissance)
      .addGrid(lr.fitIntercept)
      .build()    // ici, l'estimateur est simplement la régression linéaire.

    // Un TrainValidationSplit nécessite un estimateur, un ensemble d'estimateurs ParamMaps et un évaluateur.
    val trainValidationSplit = new TrainValidationSplit()
        .setEstimator(lr)
        .setEvaluator(new RegressionEvaluator)
        .setEstimatorParamMaps(paramGrid)
        // 70% des données seront utilisées pour le test et les 30% restants pour la validation.
        .setTrainRatio(0.7)    // Run train validation split, and choose the best set of parameters.

    // Prédictions sur les données de test:
    // model est le modèle avec la meilleure combinaison de paramètres.
    val model = trainValidationSplit.fit(trainingData)

    model.transform(testData)
      .select("features", "label", "prediction")
      .show()    // evaluateur

    // Calcul de la précision du modèle
    val bineval = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .evaluate(model.transform(testData))


    // Afficher un score permettant d’évaluer la pertinence du modèle sur les données test.
    model.transform(testData).groupBy("label", "prediction").count.show()
    println("La précision de notre modèle est: %.3".format(bineval))


    // C) Sauvegarde du modèle
    model.save("/Users/tg104460/Downloads/SPARK/LassoModel")



  }


}

