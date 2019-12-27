package io.yaochi.recommendation.model.encoder

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor

class SecondOrderEncoder(batchSize: Int,
                         nFields: Int,
                         embeddingDim: Int) {
  private val module = buildModule()

  def forward(input: Tensor[Float]): Tensor[Float] = {
    module.forward(input).toTensor
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    module.backward(input, gradOutput).toTensor[Float]
  }

  private def buildModule(): Sequential[Float] = {
    val squareSumTerms = Sequential[Float]()
      .add(Sum[Float](dimension = 2))
      .add(Power[Float](2))

    val secondOrderTerms = Sequential[Float]()
      .add(Power[Float](2))
      .add(Sum[Float](dimension = 2))

    Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields, embeddingDim), Some(false)))
      .add(DuplicateTable[Float]().add(squareSumTerms).add(secondOrderTerms))
      .add(CSubTable())
      .add(Mean[Float](dimension = 2, squeeze = false))
      .add(MulConstant(0.5))
  }
}

object SecondOrderEncoder {
  def apply(batchSize: Int,
            nFields: Int,
            embeddingDim: Int): SecondOrderEncoder = new SecondOrderEncoder(batchSize, nFields, embeddingDim)
}
