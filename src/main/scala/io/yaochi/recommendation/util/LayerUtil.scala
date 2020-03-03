package io.yaochi.recommendation.util

import com.intel.analytics.bigdl.nn.{CAdd, Linear}
import com.intel.analytics.bigdl.tensor.Tensor

object LayerUtil {
  def buildLinear(inputSize: Int,
                  outputSize: Int,
                  mats: Array[Float],
                  withBias: Boolean,
                  offset: Int): Linear[Float] = {
    val weightSize = outputSize * inputSize
    val weights = new Array[Float](weightSize)
    Array.copy(mats, offset, weights, 0, weightSize)
    val weightTensor = Tensor.apply(weights, Array(outputSize, inputSize))

    if (withBias) {
      val bias = new Array[Float](outputSize)
      Array.copy(mats, offset + weightSize, bias, 0, outputSize)
      val biasTensor = Tensor.apply(bias, Array(outputSize))

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

    val biasOffset = biasTensor.storageOffset() - 1
    val biasArray = biasTensor.storage().array()
    Array.copy(mats, offset, biasArray, biasOffset, outputSize)

    biasLayer
  }
}
