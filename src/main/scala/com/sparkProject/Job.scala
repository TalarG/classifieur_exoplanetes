package com.sparkProject

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._


object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    import spark.implicits._


    /** ******************************************************************************
      *
      * TP 1
      *
      * - Set environment, InteliJ, submit jobs to Spark
      * - Load local unstructured data
      * - Word count , Map Reduce
      * *******************************************************************************/


    // ----------------- word count ------------------------

    val df_wordCount = sc.textFile("/Users/tg104460/formationBigData/SPARK/spark-2.0.1-bin-hadoop2.7/README.md")
      .flatMap { case (line: String) => line.split(" ") }
      .map { case (word: String) => (word, 1) }
      .reduceByKey { case (i: Int, j: Int) => i + j }


      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()


    /** ******************************************************************************
      *
      * TP 2 : début du projet
      *
      * *******************************************************************************/


    //Charger le fichier csv dans un dataFrame
    val df = spark.read.option("header", "true")
      .option("separator", ",")
      .option("comment", "#")
      .csv("/Users/tg104460/formationBigData/SPARK/cumulative.csv")
    //Afficher le nombre de lignes et le nombre de colonnes dans le dataFrame.
    println("number of columns", df.columns.length)
    println("number of rows", df.count)
    //Afficher le dataFrame sous forme de table.
    df.show()

    df.select("koi_time0bk", "koi_sage_err2").show(5)
    df.printSchema()

    val columns = df.columns.slice(20, 30) // select 10 columns
    df.select(columns.map(col): _*).show(50) // show first 50 lines

    df.groupBy($"koi_disposition").count().show()

    val df_cleaned = df.filter($"koi_disposition" === "CONFIRMED" || $"koi_disposition" === "FALSE POSITIVE")
    val filterDf = df.filter("koi_disposition in (\"CONFIRMED\", \"FALSE POSITIVE\")")
    df_cleaned.groupBy($"koi_eccen_err1").count().show()
    val df_cleaned2 = df_cleaned.drop($"koi_eccen_err1")

    val df_cleaned3 = df_cleaned2.drop("index", "kepid", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
      "koi_sparprov", "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs", "koi_tce_delivname",
      "koi_parm_prov", "koi_limbdark_mod", "koi_fittype", "koi_disp_prov", "koi_comment", "kepoi_name", "kepler_name",
      "koi_vet_date", "koi_pdisposition")
    df.select(columns.map(col): _*).show(50)



    //val rows = Seq.tabulate(100) { i =>
    // if (i % 2 == 0) (1, -1.0) else (i, i * -1.0)
    //}
    //val freqSingles = df_cleaned3.stat.freqItems(Seq("koi_time0", "koi_time0_err1", "koi_time0_err2", "koi_eccen", "koi_eccen_err1", "koi_eccen_err2", "koi_longp", "koi_longp_err1", "koi_longp_err2", "koi_impact"), 0.4)
    //freqSingles.show()

    // val df_cleaned4 = df_cleaned3.stat.freqItems([ "koi_time0", "koi_time0_err1", "koi_time0_err2", "koi_eccen", "koi_eccen_err1", "koi_eccen_err2", "koi_longp", "koi_longp_err1", "koi_eccen_err2", "koi_impact" ], 0.1)
    //val useless_column = df_cleaned3.columns.filter{ case (column:String) =>
    //df_cleaned3.agg(countDistinct(column)).first().getLong(0) <= 1 }
    //df.select(columns.map(col): _*).show(50)

    //val useless_column = df_cleaned3.columns.filter{ case (column:String) =>
    //  df_cleaned3.agg(countDistinct(column)).first().getLong(0) <= 1 }

    val useless = for (col <- df_cleaned3.columns if df.select(col).distinct().count() <= 1) yield col
    val df3 = df_cleaned3.drop(useless: _*)
    df.select(columns.map(col): _*).show(50)

    df3.describe("koi_impact", "koi_duration").show()
    val df4 = df3.na.fill(0.0)

    val df_labels = df4.select("rowid", "koi_disposition")
    val df_features = df4.drop("koi_disposition")
    val df_joined = df_features.join(df_labels, usingColumn = "rowid")

    def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)
    val df_newFeatures = df_joined
      .withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2"))
      .withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")

    df_newFeatures
      .coalesce(1) // optional : regroup all data in ONE partition, so that results are printed in ONE file
      // >>>> You should not do that in general, only when the data are small enough to fit in the memory of a single machine.
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("/Users/tg104460/spark/TP_parisTech/cleanedDataFrame.csv")

  }
}