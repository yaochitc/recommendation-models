package io.yaochi.recommendation.model.deepfm

import io.yaochi.recommendation.model.{RecModel, RecModelType}

class DeepFM(inputDim: Int, nFields: Int, embeddingDim: Int, fcDims: Array[Int])
  extends RecModel(RecModelType.BIAS_WEIGHT_EMBEDDING_MATS) {

  override def getMatsSize: Array[Int] = {
    val dims = Array(nFields * embeddingDim) ++ fcDims ++ Array(1)
    (1 until dims.length)
      .map(i => Array(dims(i - 1), dims(i), dims(i), 1))
      .reduce(_ ++ _)
  }

  override def getInputDim: Int = inputDim

  override def getEmbeddingDim: Int = embeddingDim

  override protected def forward(params: Map[String, Any]): Array[Float] = {
    val batchSize = params("batch_size").asInstanceOf[Int]
    val index = params("index").asInstanceOf[Array[Long]].map(_.toInt)
    val weights = params("weights").asInstanceOf[Array[Float]]
    val bias = params("bias").asInstanceOf[Array[Float]]
    val embedding = params("embedding").asInstanceOf[Array[Float]]
    val mats = params("mats").asInstanceOf[Array[Float]]

    null
  }

  override protected def backward(params: Map[String, Any]): Float = {
    val batchSize = params("batch_size").asInstanceOf[Int]
    val index = params("index").asInstanceOf[Array[Long]].map(_.toInt)
    val weights = params("weights").asInstanceOf[Array[Float]]
    val bias = params("bias").asInstanceOf[Array[Float]]
    val embedding = params("embedding").asInstanceOf[Array[Float]]
    val mats = params("mats").asInstanceOf[Array[Float]]
    val targets = params("targets").asInstanceOf[Array[Float]]

    0f
  }

}

private[deepfm] object InternalDeepFMModel {

}
