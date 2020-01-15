package io.yaochi.recommendation.util

import com.intel.analytics.bigdl.nn.{CAdd, Linear}

object BackwardUtil {
  def linearBackward(linear: Linear[Float],
                     weights: Array[Float],
                     offset: Int = 0): Int = {
    val gradWeight = linear.gradWeight
    val gradWeightArray = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset() - 1

    val gradWeightSize = gradWeight.size()
    val outputSize = gradWeightSize(0)
    val inputSize = gradWeightSize(1)
    val weightSize = outputSize * inputSize

    Array.copy(gradWeightArray, gradWeightOffset, weights, offset, weightSize)

    var curOffset = offset + weightSize

    if (linear.withBias) {
      val gradBias = linear.gradBias
      val gradBiasArray = gradBias.storage().array()
      val gradBiasOffset = gradBias.storageOffset() - 1
      Array.copy(gradBiasArray, gradBiasOffset, weights, offset + weightSize, outputSize)
      curOffset += outputSize
    }
    curOffset
  }

  def biasBackward(bias: CAdd[Float],
                   weights: Array[Float],
                   offset: Int): Int = {
    val gradBias = bias.gradBias
    val gradArray = gradBias.storage().array()
    val gradOffset = gradBias.storageOffset() - 1
    val outputSize = gradBias.size(1)

    Array.copy(gradArray, gradOffset, weights, offset, outputSize)
    offset + outputSize
  }

}
