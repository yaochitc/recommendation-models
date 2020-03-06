package io.yaochi.recommendation.util

import com.intel.analytics.bigdl.tensor.Tensor

object GradUtil {

  def weightsGrad(weights: Array[Float],
                  weightGradTensor: Tensor[Float]): Unit = {
    val gradWeightArray = weightGradTensor.storage().array()
    val gradWeightOffset = weightGradTensor.storageOffset() - 1

    Array.copy(gradWeightArray, gradWeightOffset, weights, 0, weights.length)
  }

  def biasGrad(bias: Array[Float],
               biasGradTensor: Tensor[Float]): Unit = {
    val gradBiasArray = biasGradTensor.storage().array()
    val gradBiasOffset = biasGradTensor.storageOffset() - 1

    Array.copy(gradBiasArray, gradBiasOffset, bias, 0, bias.length)
  }

  def embeddingGrad(embedding: Array[Float],
                    gradTensors: Array[Tensor[Float]]): Unit = {
    val embeddingGradTensor = Tensor[Float]().resizeAs(gradTensors(0))
    for (gradTensor <- gradTensors) {
      embeddingGradTensor.add(gradTensor)
    }

    val gradEmbeddingArray = embeddingGradTensor.storage().array()
    val gradEmbeddingOffset = embeddingGradTensor.storageOffset() - 1

    Array.copy(gradEmbeddingArray, gradEmbeddingOffset, embedding, 0, embedding.length)
  }

  def embeddingGrad(embedding: Array[Float],
                    gradTensor: Tensor[Float]): Unit = {
    val gradEmbeddingArray = gradTensor.storage().array()
    val gradEmbeddingOffset = gradTensor.storageOffset() - 1

    Array.copy(gradEmbeddingArray, gradEmbeddingOffset, embedding, 0, embedding.length)
  }
}
