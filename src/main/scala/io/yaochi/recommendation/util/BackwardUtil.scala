package io.yaochi.recommendation.util

import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.tensor.Tensor

object BackwardUtil {
  def linearBackward(linear: Linear[Float],
                     weights: Array[Float],
                     offset: Int): Int = {
    val gradWeight = linear.gradWeight

    val gradWeightSize = gradWeight.size()
    val outputSize = gradWeightSize(0)
    val inputSize = gradWeightSize(1)

    var curOffset = offset
    for (i <- 0 until outputSize; j <- 0 until inputSize) {
      weights(curOffset + i * inputSize + j) = gradWeight.valueAt(i + 1, j + 1)
    }
    curOffset += outputSize * inputSize

    if (linear.withBias) {
      val gradBias = linear.gradBias
      for (i <- 0 until outputSize) {
        weights(curOffset + i) = gradBias.valueAt(i + 1)
      }
      curOffset += outputSize
    }
    curOffset
  }

  def weightsBackward(weights: Array[Float],
                      weightGradTensor: Tensor[Float]
                     ): Unit = {
    for (i <- 0 until weightGradTensor.nElement()) {
      weights(i) = weightGradTensor.valueAt(i + 1)
    }
  }

  def biasBackward(bias: Array[Float],
                   biasGradTensor: Tensor[Float]): Unit = {
    for (i <- bias.indices) {
      bias(i) = biasGradTensor.valueAt(i + 1)
    }
  }

  def embeddingBackward(embedding: Array[Float],
                        gradTensors: Array[Tensor[Float]]): Unit = {
    for (i <- embedding.indices) {
      embedding(i) = 0
    }

    for (gradTensor <- gradTensors) {
      for (i <- 0 until gradTensor.nElement()) {
        embedding(i) = gradTensor.valueAt(i + 1)
      }
    }
  }
}
