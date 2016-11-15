# classifieur_exoplanetes
# ligne de commande pour lancer le projet:
+ ./spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir="/tmp"  --class com.sparkProject.JobML /Users/tg104460/Downloads/tp_spark/target/scala-2.11/tp_spark-assembly-1.0.jar>output

# Le fichier de sortie est "output"

#Resultat de l'execution
#



+-----+----------+-----+

|label|prediction|count|

+-----+----------+-----+

|  1.0|       1.0|   52|

|  0.0|       1.0|    5|

|  1.0|       0.0|    5|

|  0.0|       0.0|  179|

+-----+----------+-----+


#La précision de notre modèle est: 0,943


