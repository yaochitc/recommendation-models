package io.yaochi.recommendation.util

import com.intel.analytics.bigdl.nn.Linear
import com.intel.analytics.bigdl.tensor.Tensor

object LayerUtil {
  def buildLinear(inputSize: Int,
                  outputSize: Int,
                  mats: Array[Float],
                  offset: Int): Linear[Float] = {
    val weightTensor = Tensor[Float](outputSize, inputSize)
    val biasTensor = Tensor[Float](outputSize)

    for (i <- 0 until outputSize; j <- 0 until inputSize) {
      weightTensor.setValue(i + 1, j + 1, mats(i * inputSize + j))
    }

    for (i <- 0 until outputSize) {
      biasTensor.setValue(i + 1, mats(inputSize * outputSize + i))
    }

    Linear[Float](inputSize, outputSize, initWeight = weightTensor, initBias = biasTensor)
  }
}
