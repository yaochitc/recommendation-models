package io.yaochi.recommendation.model.xdeepfm

import io.yaochi.recommendation.model.{RecModel, RecModelType}

class XDeepFM(inputDim: Int, nFields: Int, embeddingDim: Int, fcDims: Array[Int], cinDims: Array[Int])
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
    null
  }

  override protected def backward(params: Map[String, Any]): Float = {
    0f
  }
}
