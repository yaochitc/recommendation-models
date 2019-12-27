package io.yaochi.recommendation.util

import com.intel.analytics.bigdl.tensor.Tensor

object BackwardUtil {
  def weightsBackward(weights: Array[Float],
                      bias: Array[Float],
                      weightGradTensor: Tensor[Float],
                      biasGradTensor: Tensor[Float]
                     ): Unit = {
    for (i <- 0 until weightGradTensor.nElement()) {
      weights(i) = weightGradTensor.valueAt(i + 1)
    }

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