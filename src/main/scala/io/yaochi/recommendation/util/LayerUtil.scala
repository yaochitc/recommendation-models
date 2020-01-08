package io.yaochi.recommendation.util

import com.intel.analytics.bigdl.nn.{CAdd, Linear}
import com.intel.analytics.bigdl.tensor.Tensor

object LayerUtil {
  def buildLinear(inputSize: Int,
                  outputSize: Int,
                  mats: Array[Float],
                  withBias: Boolean,
                  offset: Int): Linear[Float] = {
    val weightTensor = Tensor[Float](outputSize, inputSize)

    for (i <- 0 until outputSize; j <- 0 until inputSize) {
      weightTensor.setValue(i + 1, j + 1, mats(offset + i * inputSize + j))
    }

    if (withBias) {
      val biasTensor = Tensor[Float](outputSize)
      for (i <- 0 until outputSize) {
        biasTensor.setValue(i + 1, mats(offset + inputSize * outputSize + i))
      }

      Linear[Float](inputSize, outputSize, initWeight = weightTensor, initBias = biasTensor)
    } else {
      Linear[Float](inputSize, outputSize, initWeight = weightTensor, withBias = false)
    }
  }

  def buildBiasLayer(outputSize: Int,
                     mats: Array[Float],
                     offset: Int): CAdd[Float] = {
    val biasLayer = CAdd[Float](Array(outputSize))

    val biasTensor = biasLayer.bias
    for (i <- 0 until outputSize) {
      biasTensor.setValue(i + 1, mats(offset + i))
    }

    biasLayer
  }
}
