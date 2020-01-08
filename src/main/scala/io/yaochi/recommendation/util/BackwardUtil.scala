package io.yaochi.recommendation.util

import com.intel.analytics.bigdl.nn.{CAdd, Linear}

object BackwardUtil {
  def linearBackward(linear: Linear[Float],
                     weights: Array[Float],
                     offset: Int = 0): Int = {
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

  def biasBackward(bias: CAdd[Float],
                   weights: Array[Float],
                   offset:Int):Int = {
    val gradBias = bias.gradBias
    val outputSize = gradBias.size(1)

    for (i <- 0 until outputSize) {
      weights(offset + i) = gradBias.valueAt(i + 1)
    }
    offset + outputSize
  }

}
