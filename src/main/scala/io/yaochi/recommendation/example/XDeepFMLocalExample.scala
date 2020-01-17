package io.yaochi.recommendation.example

import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.core.metric.AUC
import io.yaochi.recommendation.model.ParRecModel
import io.yaochi.recommendation.model.xdeepfm.XDeepFM
import io.yaochi.recommendation.optim.AsyncAdam
import org.apache.spark.{SparkConf, SparkContext}

object XDeepFMLocalExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "")
    val batchSize = params.getOrElse("batchSize", "100").toInt
    val stepSize = params.getOrElse("stepSize", "0.0025").toFloat
    val inputDim = params.getOrElse("inputDim", "-1").toInt
    val nFields = params.getOrElse("nFields", "-1").toInt
    val embeddingDim = params.getOrElse("embeddingDim", "-1").toInt
    val fcDims = params.getOrElse("fcDims", "").split(",").map(_.toInt)
    val cinDims = params.getOrElse("cinDims", "").split(",").map(_.toInt)

    val conf = new SparkConf()
    conf.setMaster("local[1]")
    conf.setAppName("local torch example")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    PSContext.getOrCreate(sc)
    val data = sc.textFile(input)

    val optim = new AsyncAdam(stepSize)
    val model = new ParRecModel(optim, new XDeepFM(inputDim, nFields, embeddingDim, fcDims, cinDims))
    model.init()

    for (epoch <- 1 to 50) {
      val epochStartTime = System.currentTimeMillis()
      val (lossSum, size) = data.mapPartitions {
        iterator =>
          iterator.sliding(batchSize, batchSize)
            .map(batch => (model.optimize(batch.toArray), batch.length))
      }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))

      val scores = data.mapPartitions {
        iterator =>
          iterator.sliding(batchSize, batchSize)
            .map(batch => model.predict(batch.toArray))
            .flatMap(f => f._1.zip(f._2))
            .map(f => (f._1.toDouble, f._2.toDouble))
      }

      val auc = new AUC().calculate(scores)

      val epochTime = System.currentTimeMillis() - epochStartTime
      println(s"epoch=$epoch loss=${lossSum / size} auc=$auc time=${epochTime}ms")
    }

    PSContext.stop()
    sc.stop()
  }
}
